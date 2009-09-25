"""
cargo/labor/worker.py

Host individual condor jobs.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.labor.worker import main

    raise SystemExit(main())

import os
import logging

from time import sleep
from socket import getfqdn
from sqlalchemy import (
    select,
    bindparam,
    outerjoin,
    )
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.functions import (
    count,
    random,
    )
from cargo.log import get_logger
from cargo.flags import (
    Flag,
    FlagSet,
    with_flags_parsed,
    )
from cargo.sugar import run_once
from cargo.labor.storage import (
    JobRecord,
    LaborSession,
    WorkerRecord,
    CondorWorkerRecord,
    )
from cargo.errors import print_ignored_error

log = get_logger(__name__, level = None)

class ModuleFlags(FlagSet):
    """
    Flags that apply to this module.
    """

    flag_set_title = "Worker Configuration"

    poll_work_flag = \
        Flag(
            "--poll-work",
            type    = int,
            default = 16,
            metavar = "SECONDS",
            help    = "poll for work with period SECONDS [%default]",
            )

flags = ModuleFlags.given

class NoWorkError(Exception):
    """
    No work is available.
    """

    pass

def get_worker():
    """
    Create and return a record for this worker.
    """

    try:
        cluster = os.environ["CONDOR_CLUSTER"]
        process = os.environ["CONDOR_PROCESS"]
    except KeyError:
        worker = WorkerRecord()
    else:
        worker = \
            CondorWorkerRecord(
                cluster = cluster,
                process = process,
                )

    worker.fqdn = getfqdn()

    return worker

def find_work(session, worker):
    """
    Find, acquire, and return a unit of work.
    """

    # some SQL (which we generate via SA to stay kinda portable)
    statement  =                                                \
        WorkerRecord.__table__                                  \
        .update()                                               \
        .where(WorkerRecord.uuid == worker.uuid)                \
        .values(
            job_uuid =                                          \
                select(
                    (JobRecord.uuid,),
                    JobRecord.completed == False,
                    from_obj = (JobRecord.__table__.outerjoin(WorkerRecord.job),),
                    group_by = JobRecord.uuid,
                    order_by = (count(WorkerRecord.uuid), random()),
                    limit    = 1,
                    ),
            )

    session.connection().execute(statement)
    session.expire(worker)
    session.commit()

    return worker.job

def labor_loop(session, worker):
    """
    Labor until death.
    """

    while True:
        job_record = find_work(session, worker)

        if job_record is None:
            log.note("no work available; sleeping")

            sleep(flags.poll_work)
        else:
            job_record.work.run()

            job_record.completed = True
            worker.job           = None

            session.commit()

def main_loop():
    """
    Labor, reconnecting to the database when necessary.
    """

    WAIT_TO_RECONNECT = 32
    worker            = get_worker()

    try:
        while True:
            try:
                session = LaborSession()
                worker  = session.merge(worker)

                session.commit()

                labor_loop(session, worker)
            except OperationalError, error:
                log.warning("operational error in database layer:\n%s", error)

            sleep(WAIT_TO_RECONNECT)
    finally:
        try:
            session.delete(worker)
            session.commit()
        except:
            print_ignored_error()

@with_flags_parsed()
def main(positional):
    """
    Application entry point.
    """

    # logging setup
    get_logger("sqlalchemy.engine").setLevel(logging.INFO)

    # worker body
    main_loop()


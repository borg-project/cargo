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
import numpy

from time import sleep
from uuid import (
    UUID,
    uuid4,
    )
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
from cargo.sql.alchemy import SQL_Engines
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.sugar import run_once
from cargo.labor.storage import (
    JobRecord,
    LaborSession,
    WorkerRecord,
    CondorWorkerRecord,
    labor_connect,
    )
from cargo.errors import print_ignored_error

log          = get_logger(__name__, level = None)
script_flags = \
    Flags(
        "Worker Configuration",
        Flag(
            "--poll-mu",
            type    = float,
            default = -1,
            metavar = "SECONDS",
            help    = "poll with mean SECONDS [%default]",
            ),
        Flag(
            "--poll-sigma",
            type    = float,
            default = 32.0,
            metavar = "SECONDS",
            help    = "poll with standard deviation SECONDS [%default]",
            ),
        Flag(
            "--worker-uuid",
            default = str(uuid4()),
            metavar = "UUID",
            help    = "this worker is UUID [%default]",
            ),
        Flag(
            "--job-set-uuid",
            metavar = "UUID",
            help    = "run only jobs in set UUID",
            ),
        Flag(
            "--work-chattily",
            action  = "store_true",
            help    = "enable worker verbosity on startup",
            ),
        )

class NoWorkError(Exception):
    """
    No work is available.
    """

    pass

def get_worker():
    """
    Create and return a record for this worker.
    """

    uuid = UUID(script_flags.given.worker_uuid)

    try:
        cluster = os.environ["CONDOR_CLUSTER"]
        process = os.environ["CONDOR_PROCESS"]
    except KeyError:
        worker = WorkerRecord(uuid = uuid)
    else:
        worker = \
            CondorWorkerRecord(
                uuid    = uuid,
                cluster = cluster,
                process = process,
                )

    worker.fqdn = getfqdn()

    return worker

def find_work(session, worker):
    """
    Find, acquire, and return a unit of work.
    """

    # some SQL (which we generate via SA to stay portable)
    if script_flags.given.job_set_uuid:
        job_filter = (JobRecord.completed == False) & (JobRecord.job_set_uuid == UUID(script_flags.given.job_set_uuid))
    else:
        job_filter = JobRecord.completed == False

    statement =                                                 \
        WorkerRecord.__table__                                  \
        .update()                                               \
        .where(WorkerRecord.uuid == worker.uuid)                \
        .values(
            job_uuid =                                          \
                select(
                    (JobRecord.uuid,),
                    job_filter,
                    from_obj = (JobRecord.__table__.outerjoin(WorkerRecord.job),),
                    group_by = JobRecord.uuid,
                    order_by = (count(WorkerRecord.uuid), random()),
                    limit    = 1,
                    ),
            )

    # grab a unit of work
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
            if script_flags.given.poll_mu >= 0:
                sleep_for = \
                    max(
                        0.0,
                        numpy.random.normal(
                            script_flags.given.poll_mu,
                            script_flags.given.poll_sigma,
                            ),
                        )

                log.note("no work available; sleeping for %.2f s", sleep_for)

                sleep(sleep_for)

                log.note("woke up")
            else:
                log.note("no work available; terminating")

                break
        else:
            log.note("working on job %s", job_record.uuid)

            job_record.work.run_with_fixture()

            job_record.completed = True
            worker.job           = None

            session.commit()

            log.note("finished job")

def main_loop():
    """
    Labor, reconnecting to the database when necessary.
    """

    WAIT_TO_RECONNECT = 32
    worker            = get_worker()

    try:
        while True:
            try:
                LaborSession.configure(bind = labor_connect())

                session = LaborSession()
                worker  = session.merge(worker)

                session.commit()

                labor_loop(session, worker)

                break
            except OperationalError, error:
                log.warning("operational error in database layer:\n%s", error)

            sleep(WAIT_TO_RECONNECT)
    finally:
        try:
            session.rollback()
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
    if script_flags.given.work_chattily:
        get_logger("sqlalchemy.engine").setLevel(logging.DEBUG)

    # worker body
    with SQL_Engines.default:
        main_loop()


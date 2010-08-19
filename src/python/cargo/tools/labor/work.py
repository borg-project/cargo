"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                   import call
    from cargo.tools.labor.work import main

    call(main)

from uuid      import (
    uuid4,
    UUID,
    )
from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__)

def get_worker(session, worker_uuid = None):
    """
    Create and return a record for this worker.
    """

    # grab the worker
    from cargo.labor.storage import (
        WorkerRecord,
        CondorWorkerRecord,
        )

    try:
        import os

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

    # update our host
    from socket import getfqdn

    worker.fqdn = getfqdn()

    return session.merge(worker)

def acquire_work(session, worker, job_set_uuid = None, max_hired = 1):
    """
    Find, acquire, and return a unit of work.
    """

    from uuid import UUID

    # some SQL
    from sqlalchemy               import (
        Integer,
        select,
        outerjoin,
        literal_column,
        )
    from sqlalchemy.sql.functions import count
    from cargo.labor.storage      import (
        JobRecord,
        WorkerRecord,
        )

    if job_set_uuid is None:
        job_filter = JobRecord.completed == False
    else:
        job_filter =                                                            \
            (JobRecord.completed == False)                                      \
            & (JobRecord.job_set_uuid == UUID(job_set_uuid))

    statement =                                                 \
        WorkerRecord.__table__                                  \
        .update()                                               \
        .where(WorkerRecord.uuid == worker.uuid)                \
        .values(
            job_uuid =                                          \
                select(
                    ("uuid",),
                    literal_column("hired", Integer) < max_hired,
                    select(
                        (
                            JobRecord.uuid,
                            count(WorkerRecord.uuid).label("hired"),
                            ),
                        job_filter,
                        from_obj = (JobRecord.__table__.outerjoin(WorkerRecord.job),),
                        group_by = JobRecord.uuid,
                        order_by = JobRecord.uuid,
                        )                                                              \
                        .alias("uuid_by_hired"),
                    limit    = 1,
                    ),
            )

    session.commit()

    # prevent two workers from grabbing the same unit
    from cargo.sql.alchemy import lock_table

    lock_table(session.connection(), WorkerRecord.__tablename__, "exclusive")

    # grab a unit of work
    session.execute(statement)
    session.commit()
    session.expire(worker)

def main_loop(job_set_uuid = None, worker_uuid = None, max_hired = 1):
    """
    Labor, reconnecting to the database when necessary.
    """

    from cargo.errors        import Raised
    from cargo.labor.storage import LaborSession
    from cargo.labor.storage import labor_connect

    LaborSession.configure(bind = labor_connect())

    with LaborSession() as session:
        try:
            worker = get_worker(session, worker_uuid)

            while True:
                acquire_work(session, worker, job_set_uuid, max_hired)

                if worker.job is None:
                    log.note("no work available; terminating")

                    break
                else:
                    # do the work
                    log.note("working on job %s", worker.job.uuid)

                    work = worker.job.work

                    session.commit()

                    work.run_with_fixture()

                    # mark it as done
                    log.note("finished job")

                    worker.job.completed = True
                    worker.job           = None

                    session.commit()
        except KeyboardInterrupt:
            raised = Raised()

            try:
                session.rollback()
                session.delete(worker)
                session.commit()
            except:
                Raised().print_ignored()

            raised.re_raise()

@annotations(
    job_set_uuid = ("job set on which to work", "positional", None, UUID),
    worker_uuid  = ("worker identifier"       , "positional", None, UUID),
    verbose      = ("be noisier?"             , "flag"      , "v"),
    max_hired    = ("hiring ceiling"          , "option"    , None, int),
    )
def main(job_set_uuid = None, worker_uuid = uuid4(), verbose = False, max_hired = 1):
    """
    Application entry point.
    """

    # logging setup
    if verbose:
        from cargo.log import enable_default_logging

        enable_default_logging()

        get_logger("cargo.sql.alchemy",  level = "DEBUG")
        get_logger("cargo.labor.worker", level = "DEBUG")
        get_logger("sqlalchemy.engine",  level = "DEBUG")

    # worker body
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        main_loop(job_set_uuid, worker_uuid, max_hired)


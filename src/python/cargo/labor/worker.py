"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.labor.worker import main

    raise SystemExit(main())

from cargo.log           import get_logger
from cargo.flags         import (
    Flag,
    Flags,
    with_flags_parsed,
    )

log          = get_logger(__name__)
script_flags = \
    Flags(
        "Worker Configuration",
        Flag(
            "--worker-uuid",
            default = None,
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
        Flag(
            "--max-hired",
            default = 1,
            type    = int,
            metavar = "INT",
            help    = "allow at most INT workers on a job [%default]",
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

    # assign ourselves a uuid
    from uuid import (
        UUID,
        uuid4,
        )

    if script_flags.given.worker_uuid is None:
        uuid = uuid4()
    else:
        uuid = UUID(script_flags.given.worker_uuid)

    # grab the worker
    import os

    from cargo.labor.storage import (
        WorkerRecord,
        CondorWorkerRecord,
        )

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

    # update our host
    from socket import getfqdn

    worker.fqdn = getfqdn()

    return worker

def acquire_work(session, worker):
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

    if script_flags.given.job_set_uuid:
        job_filter =                                                            \
            (JobRecord.completed == False)                                      \
            & (JobRecord.job_set_uuid == UUID(script_flags.given.job_set_uuid))
    else:
        job_filter = JobRecord.completed == False

    statement =                                                 \
        WorkerRecord.__table__                                  \
        .update()                                               \
        .where(WorkerRecord.uuid == worker.uuid)                \
        .values(
            job_uuid =                                          \
                select(
                    ("uuid",),
                    literal_column("hired", Integer) < script_flags.given.max_hired,
                    select(
                        (
                            JobRecord.uuid,
                            count(WorkerRecord.uuid).label("hired"),
                            ),
                        job_filter,
                        from_obj = (JobRecord.__table__.outerjoin(WorkerRecord.job),),
                        group_by = JobRecord.uuid,
                        order_by = ("hired", JobRecord.uuid),
                        )                                                              \
                        .alias("uuid_by_hired"),
                    limit    = 1,
                    ),
            )

    session.commit()

    # prevent two workers from grabbing the same unit
    from cargo.sql.alchemy import lock_table

    lock_table(session.connection().engine, WorkerRecord.__tablename__, "exclusive")

    # grab a unit of work
    session.execute(statement)
    session.expire(worker)
    session.commit()

def labor_loop(session, worker):
    """
    Labor until death.
    """

    while True:
        acquire_work(session, worker)

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
            worker.job.completed = True
            worker.job           = None

            session.commit()

            log.note("finished job")

def main_loop():
    """
    Labor, reconnecting to the database when necessary.
    """

    from time                import sleep
    from cargo.errors        import Raised
    from cargo.labor.storage import LaborSession

    WAIT_TO_RECONNECT = 32
    worker            = get_worker()

    try:
        from sqlalchemy.exc      import OperationalError
        from cargo.labor.storage import labor_connect

        while True:
            try:
                LaborSession.configure(bind = labor_connect())

                with LaborSession() as session:
                    worker = session.merge(worker)

                    session.commit()

                    labor_loop(session, worker)

                break
            except OperationalError, error:
                # FIXME "exception" log level?
                log.warning("operational error in database layer:\n%s", error)

            log.note("sleeping %.2fs until reconnection attempt", WAIT_TO_RECONNECT)

            sleep(WAIT_TO_RECONNECT)
    finally:
        try:
            with LaborSession() as session:
                session.rollback()
                session.delete(worker)
                session.commit()
        except:
            Raised().print_ignored()

@with_flags_parsed()
def main(positional):
    """
    Application entry point.
    """

    # logging setup
    if script_flags.given.work_chattily:
        from cargo.log import enable_default_logging

        enable_default_logging()

        get_logger("cargo.labor.worker", level = "DEBUG")
        get_logger("sqlalchemy.engine",  level = "DEBUG")

    # worker body
    from cargo.sql.alchemy import SQL_Engines

    with SQL_Engines.default:
        main_loop()


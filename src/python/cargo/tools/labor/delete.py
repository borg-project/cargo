"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                     import call
    from cargo.tools.labor.delete import main

    call(main)

from uuid        import UUID
from plac        import annotations
from cargo.log   import get_logger
from cargo       import defaults

log = get_logger(__name__, level = "NOTE")

def delete_labor(session, job_set_uuid):
    """
    Delete a job set and everything associated with it.
    """

    from cargo.labor import (
        JobRecord,
        JobRecordSet,
        WorkerRecord,
        )

    job_records = session.query(JobRecord).filter(JobRecord.job_set_uuid == job_set_uuid)

    session                                                                       \
        .query(WorkerRecord)                                                      \
        .filter(WorkerRecord.job_uuid.in_(job_records.from_self(JobRecord.uuid))) \
        .update({"job_uuid" : None}, synchronize_session = False)

    job_records.delete()

    session                                        \
        .query(JobRecordSet)                       \
        .filter(JobRecordSet.uuid == job_set_uuid) \
        .delete()

    session.commit()

@annotations(
    job_set_uuid = ("job set on which to work", "positional", None, UUID),
    url          = ("labor database URL"      , "option")   ,
    )
def main(job_set_uuid, url = defaults.labor_url):
    """
    Delete an entire set of jobs.
    """

    # set up logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    get_logger("sqlalchemy.engine", level = "DEBUG")

    # connect to the database
    from cargo.sql.alchemy import (
        make_engine,
        make_session,
        )

    Session = make_session(bind = make_engine(url))

    with Session() as session:
        delete_labor(session, job_set_uuid)


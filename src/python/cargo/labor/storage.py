"""
cargo/labor/storage.py

Store and retrieve labor records.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid import uuid4
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    PickleType,
    ForeignKey,
    )
from sqlalchemy.orm import relation
from cargo.log import get_logger
from cargo.sql.alchemy import (
    SQL_Base,
    SQL_UUID,
    SQL_Session,
    )
from cargo.flags import (
    Flag,
    Flags,
    )

log   = get_logger(__name__, level = None)
flags = \
    Flags(
        "Labor Storage Configuration",
        Flag(
            "--labor-database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use labor DATABASE [%default]",
            ),
        )

class LaborSession(SQL_Session):
    """
    Use labor storage defaults for a new session.
    """

    def __init__(self):
        """
        Initialize.
        """

        SQL_Session.__init__(self, database = flags.given.labor_database)

class JobRecordSet(SQL_Base):
    """
    Related set of job records.
    """

    __tablename__ = "job_sets"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    name = Column(String)

class JobRecord(SQL_Base):
    """
    Stored record of a unit of work.
    """

    __tablename__ = "jobs"

    uuid         = Column(SQL_UUID, primary_key = True, default = uuid4)
    job_set_uuid = Column(SQL_UUID, ForeignKey("job_sets.uuid"))
    completed    = Column(Boolean)
    work         = Column(PickleType)

    job_set = relation(JobRecordSet)

class WorkerRecord(SQL_Base):
    """
    Stored record of a worker.
    """

    __tablename__ = "workers"

    uuid     = Column(SQL_UUID, primary_key = True, default = uuid4)
    type     = Column(String)
    fqdn     = Column(String)
    job_uuid = Column(SQL_UUID, ForeignKey("jobs.uuid"))

    job = relation(JobRecord)

    __mapper_args__ = {"polymorphic_on": type}

class CondorWorkerRecord(WorkerRecord):
    """
    Stored record of a condor worker process.
    """

    __tablename__   = "condor_workers"
    __mapper_args__ = {"polymorphic_identity": "condor"}

    uuid    = Column(
        SQL_UUID,
        ForeignKey("workers.uuid"),
        default     = uuid4,
        primary_key = True,
        )
    cluster = Column(Integer)
    process = Column(Integer)

def outsource(jobs, name = None):
    """
    Appropriately outsource a set of jobs.
    """

    session = LaborSession()
    job_set = JobRecordSet(name = name)

    for job in jobs:
        job_record = \
            JobRecord(
                job_set   = job_set,
                completed = False,
                work      = job,
                )

        session.add(job_record)

    session.add(job_set)
    session.commit()
    session.close()


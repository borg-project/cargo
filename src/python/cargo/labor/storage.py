"""
cargo/labor/storage.py

Store and retrieve labor records.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid                       import uuid4
from contextlib                 import closing
from sqlalchemy                 import (
    Column,
    String,
    Boolean,
    Integer,
    PickleType,
    ForeignKey,
    )
from sqlalchemy.orm             import (
    sessionmaker,
    relationship,
    )
from sqlalchemy.ext.declarative import declarative_base
from cargo.log                  import get_logger
from cargo.sql.alchemy          import (
    SQL_UUID,
    SQL_Engines,
    )
from cargo.flags                import (
    Flag,
    Flags,
    )
from cargo.labor.jobs           import Jobs

log          = get_logger(__name__)
LaborBase    = declarative_base()
LaborSession = sessionmaker()
module_flags = \
    Flags(
        "Labor Storage Configuration",
        Flag(
            "--labor-database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use labor DATABASE [%default]",
            ),
        Flag(
            "--outsource-jobs",
            action  = "store_true",
            help    = "outsource labor to workers",
            ),
        )

class JobRecordSet(LaborBase):
    """
    Related set of job records.
    """

    __tablename__ = "job_sets"

    uuid = Column(SQL_UUID, primary_key = True, default = uuid4)
    name = Column(String)

class JobRecord(LaborBase):
    """
    Stored record of a unit of work.
    """

    __tablename__ = "jobs"

    uuid         = Column(SQL_UUID, primary_key = True, default = uuid4)
    job_set_uuid = Column(SQL_UUID, ForeignKey("job_sets.uuid"))
    completed    = Column(Boolean)
    work         = Column(PickleType(mutable = False))

    job_set = relationship(JobRecordSet)

class WorkerRecord(LaborBase):
    """
    Stored record of a worker.
    """

    __tablename__ = "workers"

    uuid     = Column(SQL_UUID, primary_key = True, default = uuid4)
    type     = Column(String)
    fqdn     = Column(String)
    job_uuid = Column(SQL_UUID, ForeignKey("jobs.uuid"))

    job = relationship(JobRecord)

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

def outsource_or_run(jobs, name = None, flags = module_flags.given):
    """
    Outsource or run a set of jobs based on flags.
    """

    jobs    = list(jobs)
    Session = sessionmaker()

    if flags.outsource_jobs:
        Session.configure(bind = labor_connect(flags = flags))

        outsource(jobs, name, Session = Session)
    else:
        log.note("running %i jobs", len(jobs))

        Jobs(jobs).run()

def outsource(jobs, name = None, Session = LaborSession):
    """
    Appropriately outsource a set of jobs.
    """

    CHUNK_SIZE = 8192
    njobs      = len(jobs)
    session    = Session()

    with closing(session):
        # create the job set
        job_set = JobRecordSet(name = name)

        session.add(job_set)
        session.flush()

        log.note("inserting %i jobs into set %s", njobs, job_set.uuid)

        # insert the jobs
        ninserted = 0

        while ninserted < njobs:
            chunk = jobs[ninserted:ninserted + CHUNK_SIZE]

            session.connection().execute(
                JobRecord.__table__.insert(),
                [
                    {
                        "job_set_uuid": job_set.uuid,
                        "completed":    False,
                        "work":         job,
                        }                             \
                    for job in chunk
                    ]
                )

            ninserted += CHUNK_SIZE
            ninserted  = min(ninserted, len(jobs))

            log.note(
                "inserted %i jobs so far (%i%%)",
                ninserted,
                ninserted * 100.0 / njobs,
                )

        session.commit()

        log.note("committed job insertions")

        return job_set.uuid

def labor_connect(engines = SQL_Engines.default, flags = module_flags.given):
    """
    Connect to acridid storage.
    """

    flags  = module_flags.merged(flags)
    engine = engines.get(flags.labor_database)

    LaborBase.metadata.create_all(engine)

    return engine


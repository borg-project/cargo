"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from uuid                       import uuid4
from sqlalchemy                 import (
    Column,
    String,
    Boolean,
    Integer,
    PickleType,
    ForeignKey,
    )
from sqlalchemy.orm             import relationship
from sqlalchemy.ext.declarative import declarative_base
from cargo.log                  import get_logger
from cargo.sql.alchemy          import (
    SQL_UUID,
    SQL_Engines,
    )
from cargo.labor.jobs           import Jobs

log          = get_logger(__name__, level = "NOTE")
LaborBase    = declarative_base()
LaborSession = make_session()

def outsource_or_run(raw_jobs, outsource, name = None, url = defaults.labor_url):
    """
    Outsource or run a set of jobs.
    """

    def filter_jobs():
        from cargo.labor.jobs import (
            Job,
            CallableJob,
            )

        for job in raw_jobs:
            if isinstance(job, Job):
                yield job
            else:
                yield CallableJob(job[0], *job[1:])

    jobs = list(filter_jobs())

    if outsource:
        from cargo.sql.alchemy import make_session

        Session = make_session(bind = labor_connect(url = url))

        with Session() as session:
            outsource_jobs(session, jobs, name)
    else:
        log.note("running %i jobs", len(jobs))

        Jobs(jobs).run()

def outsource_jobs(session, jobs, name = None):
    """
    Appropriately outsource a set of jobs.
    """

    # create the job set
    CHUNK_SIZE = 4096
    njobs      = len(jobs)
    job_set    = JobRecordSet(name = name)

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
                    "uuid"         : uuid4(),
                    "job_set_uuid" : job_set.uuid,
                    "completed"    : False,
                    "work"         : job,
                    }
                for job in chunk
                ],
            )

        ninserted += CHUNK_SIZE
        ninserted  = min(ninserted, len(jobs))

        log.note(
            "inserted %i jobs so far (%.1f%%)",
            ninserted,
            ninserted * 100.0 / njobs,
            )

    session.commit()

    log.note("committed job insertions")

    return job_set.uuid

def labor_connect(engines = SQL_Engines.default, url = defaults.labor_url):
    """
    Connect to acridid storage.
    """

    return engines.get(url)

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


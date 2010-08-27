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
from cargo                      import defaults
from cargo.log                  import get_logger
from cargo.sql.alchemy          import (
    SQL_UUID,
    SQL_Engines,
    )

log       = get_logger(__name__, level = "NOTE")
LaborBase = declarative_base()

def outsource_or_run(jobs, outsource, name = None, url = defaults.labor_url):
    """
    Outsource or run a set of jobs.
    """

    if outsource:
        from cargo.sql.alchemy import (
            make_engine,
            make_session,
            )

        Session = make_session(bind = make_engine(url))

        with Session() as session:
            outsource_jobs(session, jobs, name)
    else:
        log.note("running %i jobs immediately", len(jobs))

        for job in jobs:
            job()

def outsource_jobs(session, jobs, name = None, chunk_size = 1024):
    """
    Appropriately outsource a set of jobs.
    """

    # create the job set
    if name is None:
        from cargo.temporal import utc_now

        name = "work outsourced at %s" % utc_now()

    job_set_uuid = uuid4()
    job_set      = JobRecordSet(uuid = job_set_uuid, name = name)

    session.add(job_set)
    session.flush()

    log.note("inserting %i jobs into set %s", len(jobs), job_set.uuid)

    # insert the jobs
    count = 0

    while count < len(jobs):
        chunk  = jobs[count:count + chunk_size]
        count += len(chunk)

        session.connection().execute(
            JobRecord.__table__.insert(["uuid", "job_set_uuid", "completed", "work"]),
            [
                {
                    "uuid"         : uuid4(),
                    "job_set_uuid" : job_set_uuid,
                    "completed"    : False,
                    "work"         : job,
                    }
                for job in chunk
                ],
            )

        log.note(
            "inserted %i jobs so far (%.1f%%)",
            count,
            count * 100.0 / len(jobs),
            )

    session.commit()

    log.note("committed job insertions")

    return job_set

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

    uuid    = Column(SQL_UUID, ForeignKey("workers.uuid"), primary_key = True, default = uuid4)
    cluster = Column(Integer)
    process = Column(Integer)


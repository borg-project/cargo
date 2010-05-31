"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.labor.condor import main

    raise SystemExit(main())

import os
import datetime
import cargo.labor.storage

from collections           import namedtuple
from cargo.log             import get_logger
from cargo.flags           import (
    Flag,
    Flags,
    with_flags_parsed,
    )

log          = get_logger(__name__, level = "NOTE")
module_flags = \
    Flags(
        "Worker Configuration",
        Flag(
            "--workers",
            type    = int,
            default = 64,
            metavar = "INT",
            help    = "submit INT workers to Condor [%default]",
            ),
        Flag(
            "--matching",
            default = "InMastodon && ( Arch == \"INTEL\" ) && ( OpSys == \"LINUX\" ) && regexp(\"rhavan-.*\", ParallelSchedulingGroup)",
            metavar = "STRING",
            help    = "use nodes matching STRING [%default]",
            ),
        Flag(
            "--description",
            default = "distributed Python worker process(es)",
            metavar = "STRING",
            help    = "use cluster description STRING [%default]",
            ),
        Flag(
            "--job-set-uuid",
            metavar = "UUID",
            help    = "run only jobs in set UUID",
            ),
        )

class CondorSubmissionFile(object):
    """
    Stream output to a Condor submission file.
    """

    def __init__(self, file):
        """
        Initialize.
        """

        if isinstance(file, str):
            file = open(file, "w")

        self.file = file

    def write_blank(self, lines = 1):
        """
        Write a blank line or many.
        """

        for i in xrange(lines):
            self.file.write("\n")

    def write_pair(self, name, value):
        """
        Write a variable assignment line.
        """

        self.file.write("%s = %s\n" % (name, value))

    def write_pairs(self, **kwargs):
        """
        Write a block of variable assignment lines.
        """

        self.write_pairs_dict(kwargs)

    def write_pairs_dict(self, pairs):
        """
        Write a block of variable assignment lines.
        """

        max_len = max(len(k) for (k, _) in pairs.iteritems())

        for (name, value) in pairs.iteritems():
            self.write_pair(name.ljust(max_len), value)

    def write_environment(self, **kwargs):
        """
        Write an environment assignment line.
        """

        self.file.write("environment = \\\n")

        pairs = sorted(kwargs.items(), key = lambda (k, _): k)

        for (i, (key, value)) in enumerate(pairs):
            self.file.write("    %s=%s;" % (key, value))

            if i < len(pairs) - 1:
                self.file.write(" \\")

            self.file.write("\n")

    def write_header(self, header):
        """
        Write commented header text.
        """

        dashes = "-" * len(header)

        self.write_comment(dashes)
        self.write_comment(header.upper())
        self.write_comment(dashes)

    def write_comment(self, comment):
        """
        Write a line of commented text.
        """

        self.file.write("# %s\n" % comment)

    def write_queue(self, count):
        """
        Write a queue instruction.
        """

        self.file.write("Queue %i\n" % count)

CondorWorkerProcess = \
    namedtuple(
        "CondorWorkerProcess",
        (
            "uuid",
            "working",
            "database",
            ),
        )

class CondorSubmission(object):
    """
    Manage submission of Python workers to Condor.
    """

    class_flags = \
        Flags(
            "Condor Submission Configuration",
            Flag(
                "--condor-home",
                default = "workers-%s" % datetime.datetime.now().replace(microsecond = 0).isoformat(),
                metavar = "PATH",
                help    = "run workers under PATH [%default]",
                ),
            )

    def __init__(
        self,
        matching     = None,
        description  = "distributed Python worker process(es)",
        job_set_uuid = None,
        group        = "GRAD",
        project      = "AI_ROBOTICS",
        flags        = class_flags.given,
        ):
        """
        Initialize.
        """

        self.matching     = matching
        self.description  = description
        self.job_set_uuid = job_set_uuid
        self.group        = group
        self.project      = project
        self.flags        = self.class_flags.merged(flags)
        self.workers      = []

    def write(self, file):
        """
        Generate a submission file.
        """

        submit = CondorSubmissionFile(file)

        # write the job-matching section
        if self.matching:
            submit.write_header("node matching")
            submit.write_blank()
            submit.write_pair("requirements", self.matching)
            submit.write_blank(2)

        # write the general condor section
        import sys

        submit.write_header("condor configuration")
        submit.write_blank()
        submit.write_pairs_dict({
            "+Group"              : "\"%s\"" % self.group,
            "+Project"            : "\"%s\"" % self.project,
            "+ProjectDescription" : "\"%s\"" % self.description,
            })
        submit.write_blank()
        submit.write_pairs(
            universe     = "vanilla",
            notification = "Error",
            kill_sig     = "SIGINT",
            Log          = "condor.log",
            Error        = "condor.err",
            Output       = "condor.out",
            Input        = "/dev/null",
            Executable   = sys.executable,
            )
        submit.write_blank()
        submit.write_environment(
            CONDOR_CLUSTER  = "$(Cluster)",
            CONDOR_PROCESS  = "$(Process)",
            PATH            = os.environ.get("PATH", ""),
            PYTHONPATH      = os.environ.get("PYTHONPATH", ""),
            LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", ""),
            )
        submit.write_blank(2)

        # write the jobs section
        submit.write_header("condor jobs")
        submit.write_blank()

        for worker in self.workers:
            arguments   = \
                "-m cargo.labor.worker --labor-database %s --worker-uuid %s" % (
                    worker.database,
                    str(worker.uuid),
                    )

            if self.job_set_uuid:
                arguments += " --job-set-uuid %s" % str(self.job_set_uuid)

            submit.write_pairs(
                Initialdir = worker.working,
                Arguments  = arguments,
                )
            submit.write_queue(1)
            submit.write_blank()

    def add(self, database):
        """
        Add a worker record to the submission.

        The returned working directory will not exist before prepare().

        @return: The worker working directory.
        """

        # populate per-worker directories
        from uuid import uuid4

        uuid     = uuid4()
        job_path = os.path.join(self.flags.condor_home, str(uuid))
        process  = CondorWorkerProcess(uuid, job_path, database)

        self.workers.append(process)

        return process

    def add_many(self, nworkers, database):
        """
        Add many worker records to the submission.
        """

        return [self.add(database) for i in xrange(nworkers)]

    def prepare(self):
        """
        Generate the submission file and working directories.
        """

        for worker in self.workers:
            os.makedirs(worker.working)

        submit_path = os.path.join(self.flags.condor_home, "workers.condor")

        with open(submit_path, "w") as opened:
            self.write(opened)

    def submit(self):
        """
        Submit the job to condor.
        """

        from subprocess import check_call

        check_call([
            "/usr/bin/env",
            "condor_submit",
            os.path.join(self.flags.condor_home, "workers.condor"),
            ])

def pfork(callable, matching, *args, **kwargs):
    """
    Immediately fork a callable to a condor process.
    """

    from cargo.labor.jobs import CallableJob

    pfork_job(CallableJob(callable, *args, **kwargs), matching)

def pfork_job(job, matching):
    """
    Immediately fork a job to a condor process.
    """

    # create the job directory
    submission = CondorSubmission(matching = matching)
    process    = submission.add(database = "sqlite:///labor.sqlite")

    submission.prepare()

    # store the job
    from cargo.sql.alchemy     import make_session
    from cargo.labor.storage   import (
        outsource,
        labor_connect,
        )
    from sqlalchemy.engine.url import URL

    url     = URL("sqlite", database = os.path.join(process.working, "labor.sqlite"))
    engine  = labor_connect(flags = {"labor_database": url})
    Session = make_session(bind = engine)

    outsource([job], Session = Session)

    engine.dispose()

    # spawn the process
    submission.submit()

def submit_workers(nworkers, database, matching, description, job_set_uuid):
    """
    Fork worker submission.
    """

    submission = \
        CondorSubmission(
            matching     = matching,
            description  = description,
            job_set_uuid = job_set_uuid,
            )

    submission.add_many(nworkers, database)
    submission.prepare()
    submission.submit()

def submit_workers_default(database = None, flags = module_flags.given):
    """
    Submit workers using default settings.
    """

    flags = module_flags.merged(flags)

    if database is None:
        database = cargo.labor.storage.module_flags.given.labor_database

    submit_workers(
        flags.workers,
        database,
        flags.matching,
        flags.description,
        flags.job_set_uuid,
        )

@with_flags_parsed()
def main(positional):
    """
    Spawn condor workers.
    """

    submit_workers_default()


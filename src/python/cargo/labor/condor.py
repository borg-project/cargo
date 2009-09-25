"""
cargo/condor/spawn.py

Build and spawn condor jobs.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os
import sys
import cPickle as pickle

from uuid import uuid4
from tempfile import NamedTemporaryFile
from subprocess import check_call
from cargo.log import get_logger
from cargo.flags import (
    Flag,
    FlagSet,
    )

log = get_logger(__name__, level = None)

# FIXME clean all this stuff up

class CondorJob(object):
    """
    Manage configuration of a single condor job.

    Must remain picklable.
    """

    def __init__(self, function, *args, **kwargs):
        """
        Initialize.
        """

        self.function = function
        self.args     = args
        self.kwargs   = kwargs
        self.uuid     = uuid4()

    def run(self):
        """
        Run this job in-process.
        """

        # FIXME
#         if sys.argv[1:] != self.argv:
#             raise ValueError("process arguments do not match job arguments")

        self.function(*self.args, **self.kwargs)

class CondorJobs(CondorJob):
    """
    Manage configuration of multiple condor jobs.

    Must remain picklable.
    """

    def __init__(self, jobs = None):
        """
        Initialize.
        """

        if jobs is None:
            jobs = []

        self.jobs = jobs
        self.uuid = uuid4()

    def run(self):
        """
        Run this job in-process.
        """

        for job in self.jobs:
            job.run()

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

class CondorSubmission(object):
    """
    Manage job submission to Condor.
    """

    class Flags(FlagSet):
        """
        Flags that apply to this module.
        """

        flag_set_title = "Condor Submission Configuration"

        condor_home_flag = \
            Flag(
                "--condor-home",
                default = "jobs-%s" % uuid4(),
                metavar = "PATH",
                help    = "generate job directories under PATH [%default]",
                )
        condor_flag = \
            Flag(
                "--condor",
                action = "store_true",
                help   = "spawn condor jobs?",
                )

    def __init__(self, jobs, matching = None, argv = (), description = "cluster job", flags = Flags.given):
        """
        Initialize.
        """

        self.jobs        = jobs
        self.matching    = matching
        self.argv        = argv
        self.description = description
        self.flags       = flags

    def get_job_directory(self, job):
        """
        Get the working directory configured for C{job}.
        """

        return os.path.join(self.flags.condor_home, str(job.uuid))

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
        submit.write_header("condor configuration")
        submit.write_blank()
        submit.write_pairs_dict({
            "+Group":              "GRAD",
            "+Project":            "AI_ROBOTICS",
            "+ProjectDescription": "\"%s\"" % self.description,
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
            Executable   = "/lusr/bin/python",
            Arguments    = "-m cargo.condor.host " + " ".join(self.argv),
            )
        submit.write_blank()
        submit.write_environment(
            CONDOR_CLUSTER  = "$(Cluster)",
            CONDOR_PROCESS  = "$(Process)",
            LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", ""),
            PYTHONPATH      = os.environ.get("PYTHONPATH", ""),
            )
        submit.write_blank(2)

        # write the jobs section
        submit.write_header("condor jobs")
        submit.write_blank()

        for job in self.jobs:
            submit.write_comment("job %s" % job.uuid)
            submit.write_pair("Initialdir", self.get_job_directory(job))
            submit.write_queue(1)
            submit.write_blank()

    def run(self):
        """
        Run all jobs in-process, bypassing Condor.
        """

        for job in self.jobs:
            job.run()

    def submit(self):
        """
        Submit all jobs to Condor.
        """

        # populate per-job working directories
        for job in self.jobs:
            job_path = self.get_job_directory(job)

            os.makedirs(job_path)

            with open(os.path.join(job_path, "job.pickle"), "w") as job_file:
                pickle.dump(job, job_file)

        # generate and send the submission file
        with NamedTemporaryFile(suffix = ".condor") as temporary:
            self.write(temporary)
            temporary.flush()

            check_call(["/usr/bin/env", "condor_submit", temporary.name])

    def run_or_submit(self):
        """
        Run in-process or submit to Condor, depending on flags.
        """

        if self.flags.condor:
            self.submit()
        else:
            self.run()

def main():
    # FIXME

    # condor!
    submission = \
        CondorSubmission(
            jobs        = jobs,
            matching    = matching,
            argv        = argv,
            description = "sampling randomized heuristic solver outcome distributions",
            )

    submission.run_or_submit()


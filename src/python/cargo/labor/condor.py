"""
cargo/labor/condor.py

Spawn condor workers.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.labor.condor import main

    raise SystemExit(main())

import os
import sys
import cPickle as pickle

from uuid import uuid4
from tempfile import NamedTemporaryFile
from subprocess import check_call
from cargo.log import get_logger
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )

log          = get_logger(__name__, level = None)
script_flags = \
    Flags(
        "Worker Configuration",
        Flag(
            "--workers",
            type    = int,
            default = 64,
            metavar = "INT",
            help    = "submit INT workers to Condor [%default]",
            ),
        )

# FIXME clean this all up a bit

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

    class_flags = \
        Flags(
            "Condor Submission Configuration",
            Flag(
                "--condor-home",
                default = "jobs-%s" % uuid4(),
                metavar = "PATH",
                help    = "generate job directories under PATH [%default]",
                ),
            Flag(
                "--condor",
                action = "store_true",
                help   = "spawn condor jobs?",
                ),
            )

    def __init__(
        self,
        matching    = None,
        argv        = (),
        description = "distributed Python worker processes",
        flags       = class_flags.given,
        ):
        """
        Initialize.
        """

        self.matching    = matching
        self.argv        = argv
        self.description = description
        self.flags       = flags

    def write(self, file, job_paths):
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
            Arguments    = "-m cargo.labor.worker " + " ".join(self.argv),
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

        for job_path in job_paths:
            submit.write_pair("Initialdir", job_path)
            submit.write_queue(1)
            submit.write_blank()

    def spawn(self, nworkers = 1):
        """
        Submit all jobs to Condor.
        """

        # populate per-job working directories
        job_paths = [os.path.join(self.flags.condor_home, str(uuid4())) for i in xrange(nworkers)]

        for job_path in job_paths:
            os.makedirs(job_path)

        # generate and send the submission file
        with NamedTemporaryFile(suffix = ".condor") as temporary:
            self.write(temporary, job_paths)
            temporary.flush()

            check_call(["/usr/bin/env", "condor_submit", temporary.name])

@with_flags_parsed()
def main(positional):
    """
    Spawn condor workers.
    """

    # condor!
    # FIXME
    matching = "InMastodon && ( Arch == \"INTEL\" ) && ( OpSys == \"LINUX\" ) && regexp(\"rhavan-.*\", ParallelSchedulingGroup)"

    submission = \
        CondorSubmission(
            matching    = matching,
            # FIXME
            argv        = ("--labor-database", "postgresql://postgres@zerogravitas.csres.utexas.edu:5432/labor"),
            description = "sampling randomized heuristic solver outcome distributions",
            )

    submission.spawn(flags.workers)


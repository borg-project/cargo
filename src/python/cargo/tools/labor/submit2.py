"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.submit2 import main

    plac.call(main)

import os
import os.path
import sys
import datetime
import subprocess
import cargo

logger = cargo.get_logger(__name__, level = "NOTE")

class CondorSubmissionFile(object):
    """
    Stream output to a Condor submission file.
    """

    def __init__(self, file):
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

def submit_workers(
    workers,
    address,
    matching = None,
    description = "distributed Python worker process(es)",
    group = "GRAD",
    project = "AI_ROBOTICS",
    condor_home = "",
    ):
    # prepare the working directories
    working_paths = [os.path.join(condor_home, "%i" % i) for i in xrange(workers)]

    for working_path in working_paths:
        os.makedirs(working_path)

    submit_path = os.path.join(condor_home, "workers.condor")

    # write the submit file
    with open(submit_path, "w") as opened:
        # write the job-matching section
        submit = CondorSubmissionFile(opened)

        if matching:
            submit.write_header("node matching")
            submit.write_blank()
            submit.write_pair("requirements", matching)
            submit.write_blank(2)

        # write the general condor section
        submit.write_header("condor configuration")
        submit.write_blank()
        submit.write_pairs_dict({
            "+Group": "\"%s\"" % group,
            "+Project": "\"%s\"" % project,
            "+ProjectDescription": "\"%s\"" % description,
            })
        submit.write_blank()
        submit.write_pairs(
            universe = "vanilla",
            notification = "Error",
            kill_sig = "SIGINT",
            Log = "condor.log",
            Error = "condor.err",
            Output = "condor.out",
            Input = "/dev/null",
            Executable = os.environ.get("SHELL"),
            )
        submit.write_blank()
        submit.write_environment(
            CARGO_LOG_FILE_PREFIX = "log",
            CONDOR_CLUSTER = "$(Cluster)",
            CONDOR_PROCESS = "$(Process)",
            PATH = os.environ.get("PATH", ""),
            PYTHONPATH = os.environ.get("PYTHONPATH", ""),
            LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", ""),
            )
        submit.write_blank()

        # write the jobs section
        submit.write_header("condor jobs")
        submit.write_blank()

        for working_path in working_paths:
            arg_format = '"-c \'%s ""$0"" $@\' -m cargo.tools.labor.work2 %s"'

            submit.write_pairs(
                Initialdir = working_path,
                Arguments = arg_format % (sys.executable, address),
                )
            submit.write_queue(1)
            submit.write_blank()

    # submit the job to condor
    subprocess.check_call([
        "/usr/bin/env",
        "condor_submit",
        os.path.join(condor_home, "workers.condor"),
        ])

def default_condor_home():
    return "workers-%s" % datetime.datetime.now().replace(microsecond = 0).isoformat()

@plac.annotations(
    workers = ("number of workers to submit", "positional", None, int),
    address = ("job set on which to work", "positional"),
    matching = ("node match restriction", "option"),
    home = ("job submission directory", "option"),
    )
def submit_workers_for(
    workers,
    address,
    matching = cargo.defaults.condor_matching,
    home = default_condor_home(),
    ):
    # submit the jobs
    submit_workers(
        workers,
        address,
        matching,
        condor_home = home,
        )

    # provide a convenience symlink
    link_path = "workers-latest"

    if os.path.lexists(link_path):
        os.unlink(link_path)

    os.symlink(home, link_path)

main = submit_workers_for


"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                     import call
    from cargo.tools.labor.submit import main

    call(main)

from uuid        import UUID
from collections import namedtuple
from plac        import annotations
from cargo.log   import get_logger
from cargo       import defaults

log = get_logger(__name__, level = "NOTE")

CondorWorkerProcess = \
    namedtuple(
        "CondorWorkerProcess",
        (
            "uuid",
            "working",
            "database",
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

class CondorSubmission(object):
    """
    Manage submission of Python workers to Condor.
    """

    def __init__(
        self,
        matching     = None,
        description  = "distributed Python worker process(es)",
        job_set_uuid = None,
        group        = "GRAD",
        project      = "AI_ROBOTICS",
        condor_home  = "",
        ):
        """
        Initialize.
        """

        self.matching     = matching
        self.description  = description
        self.job_set_uuid = job_set_uuid
        self.group        = group
        self.project      = project
        self.workers      = []
        self.condor_home  = condor_home

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

        from os import environ

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
            Executable   = environ.get("SHELL"),
            )
        submit.write_blank()
        submit.write_environment(
            CONDOR_CLUSTER  = "$(Cluster)",
            CONDOR_PROCESS  = "$(Process)",
            PATH            = environ.get("PATH", ""),
            PYTHONPATH      = environ.get("PYTHONPATH", ""),
            LD_LIBRARY_PATH = environ.get("LD_LIBRARY_PATH", ""),
            )
        submit.write_blank()

        # write the jobs section
        submit.write_header("condor jobs")
        submit.write_blank()

        for worker in self.workers:
            if self.job_set_uuid:
                job_set_argument = "%s " % str(self.job_set_uuid)
            else:
                job_set_argument = ""

            arguments = \
                '-c \'%s ""$0"" $@\' -m cargo.tools.labor.work -url %s %s%s' % (
                    sys.executable,
                    worker.database,
                    job_set_argument,
                    str(worker.uuid),
                    )

            submit.write_pairs(
                Initialdir = worker.working,
                Arguments  = '"%s"' % arguments,
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
        from os.path import join
        from uuid    import uuid4

        uuid     = uuid4()
        job_path = join(self.condor_home, str(uuid))
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

        from os      import makedirs
        from os.path import join

        for worker in self.workers:
            makedirs(worker.working)

        submit_path = join(self.condor_home, "workers.condor")

        with open(submit_path, "w") as opened:
            self.write(opened)

    def submit(self):
        """
        Submit the job to condor.
        """

        from os.path    import join
        from subprocess import check_call

        check_call([
            "/usr/bin/env",
            "condor_submit",
            join(self.condor_home, "workers.condor"),
            ])

def default_condor_home():
    """
    Return the default directory for spawned jobs.
    """

    import datetime

    return "workers-%s" % datetime.datetime.now().replace(microsecond = 0).isoformat()

@annotations(
    job_set_uuid = ("job set on which to work",    "positional", None, UUID),
    workers      = ("number of workers to submit", "positional", None, int),
    url          = ("labor database URL",          "option"),
    matching     = ("node match restriction",      "option"),
    description  = ("condor job description",      "option"),
    home         = ("job submission directory",    "option"),
    )
def main(
    workers,
    job_set_uuid,
    url         = defaults.labor_url,
    matching    = defaults.condor_matching,
    description = "distributed Python worker process(es)",
    home        = default_condor_home(),
    ):
    """
    Spawn condor workers.
    """

    submission = \
        CondorSubmission(
            matching     = matching,
            description  = description,
            job_set_uuid = job_set_uuid,
            condor_home  = home,
            )

    submission.add_many(workers, url)
    submission.prepare()
    submission.submit()


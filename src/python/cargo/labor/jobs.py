"""
cargo/labor/work.py

Units of work.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from abc import abstractmethod
from uuid import uuid4
from cargo.log import get_logger
from cargo.sugar import ABC

log = get_logger(__name__, level = None)

class Job(ABC):
    """
    Interface to a unit of work.
    """

    def __init__(self, **kwargs):
        """
        Base initializer; updates object dictionary with keyword arguments.
        """

        self.__dict__.update(kwargs)

    @abstractmethod
    def run(self):
        """
        Run this job in-process.
        """

        pass

class CallableJob(Job):
    """
    Manage configuration of a single condor job.

    Must remain picklable.
    """

    def __init__(self, function, *args, **kwargs):
        """
        Initialize.
        """

        # base
        Job.__init__(self)

        # members
        self.function = function
        self.argv     = argv
        self.args     = args
        self.kwargs   = kwargs
        self.uuid     = uuid4()

    def run(self):
        """
        Run this job in-process.
        """

        self.function(*self.args, **self.kwargs)

class Jobs(Job):
    """
    Manage configuration of multiple jobs.

    Must remain picklable.
    """

    def __init__(self, jobs = None):
        """
        Initialize.
        """

        # base
        Job.__init__(self)

        # members
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


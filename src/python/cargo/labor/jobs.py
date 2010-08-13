"""
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

    def run_with_fixture(self):
        """
        Run, calling setup and teardown methods.
        """

        try:
            set_up = type(self).class_set_up
        except AttributeError:
            pass
        else:
            set_up()

        self.run()

        try:
            tear_down = type(self).class_tear_down
        except AttributeError:
            pass
        else:
            tear_down()

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
            if isinstance(job, Job):
                job.run_with_fixture()
            else:
                job[0](*job[1:])

class CallableJob(Job):
    """
    Wrap a callable as a job.
    """

    def __init__(self, callable, *args, **kwargs):
        """
        Initialize.
        """

        Job.__init__(
            self,
            callable = callable,
            args     = args,
            kwargs   = kwargs,
            )

    def run(self):
        """
        Run!
        """

        self.callable(*self.args, **self.kwargs)


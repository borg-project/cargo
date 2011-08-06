"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

from __future__ import absolute_import

import os
import sys
import time
import zlib
import socket
import random
import subprocess
import collections
import cPickle as pickle
import cargo

logger = cargo.get_logger(__name__, level = "INFO")

def send_pyobj_gz(zmq_socket, message):
    pickled = pickle.dumps(message)
    compressed = zlib.compress(pickled, 1)

    zmq_socket.send(compressed)

def recv_pyobj_gz(zmq_socket):
    compressed = zmq_socket.recv()
    decompressed = zlib.decompress(compressed)
    unpickled = pickle.loads(decompressed)

    return unpickled

class Message(object):
    """Message from a worker."""

    def __init__(self, sender):
        self.sender = sender
        self.host = socket.gethostname()
        self.pid = os.getpid()

    def make_summary(self, text):
        return "worker {0} (pid {1} on {2}) {3}".format(self.sender, self.pid, self.host, text)

class ApplyMessage(Message):
    """A worker wants a unit of work."""

    def get_summary(self):
        return self.make_summary("requested a job")

class ErrorMessage(Message):
    """An error occurred in a task."""

    def __init__(self, sender, key, description):
        Message.__init__(self, sender)

        self.key = key
        self.description = description

    def get_summary(self):
        brief = self.description.splitlines()[-1]

        return self.make_summary("encountered an error ({0})".format(brief))

class InterruptedMessage(Message):
    """A worker was interrupted."""

    def __init__(self, sender, key):
        Message.__init__(self, sender)

        self.key = key

    def get_summary(self):
        return self.make_summary("was interrupted")

class DoneMessage(Message):
    """A task was completed."""

    def __init__(self, sender, key, result):
        Message.__init__(self, sender)

        self.key = key
        self.result = result

    def get_summary(self):
        return self.make_summary("finished job {0}".format(self.key))

class Task(object):
    """One unit of distributable work."""

    def __init__(self, call, args = [], kwargs = {}):
        self.call = call
        self.args = args
        self.kwargs = kwargs
        self.key = id(self)

    def __hash__(self):
        return hash(self.key)

    def __call__(self):
        return self.call(*self.args, **self.kwargs)

    @staticmethod
    def from_request(request):
        """Build a task, if necessary."""

        if isinstance(request, Task):
            return request
        elif isinstance(request, collections.Mapping):
            return Task(**mapping)
        else:
            return Task(*request)

class TaskState(object):
    """Current state of progress on a task."""

    def __init__(self, task):
        self.task = task
        self.done = False
        self.working = set()

    def score(self):
        """Score the urgency of this task."""

        if self.done:
            return (sys.maxint, sys.maxint, random.random())
        if len(self.working) == 0:
            return (0, 0, random.random())
        else:
            return (
                len(self.working),
                max(wstate.timestamp for wstate in self.working),
                random.random(),
                )

class WorkerState(object):
    """Current state of a known worker process."""

    def __init__(self, condor_id):
        self.condor_id = condor_id
        self.assigned = None
        self.timestamp = None

    def set_done(self):
        self.assigned.working.remove(self)

        was_done = self.assigned.done

        self.assigned.done = True
        self.assigned = None

        return was_done

    def set_assigned(self, tstate):
        """Change worker state in response to assignment."""

        self.disassociate()

        self.assigned = tstate
        self.timestamp = time.time()

        self.assigned.working.add(self)

    def set_interruption(self):
        """Change worker state in response to interruption."""

        self.disassociate()

    def set_error(self):
        """Change worker state in response to error."""

        self.disassociate()

    def disassociate(self):
        """Disassociate from the current job."""

        if self.assigned is not None:
            self.assigned.working.remove(self)

            self.assigned = None

class Manager(object):
    """Manage distributed work."""

    def __init__(self, task_list, handler, rep_socket):
        """Initialize."""

        self.handler = handler
        self.rep_socket = rep_socket
        self.tstates = dict((t.key, TaskState(t)) for t in task_list)
        self.wstates = {}

    def manage(self):
        """Manage workers and tasks."""

        # prepare
        import zmq

        poller = zmq.Poller()

        poller.register(self.rep_socket, zmq.POLLIN)

        # receive
        while self.unfinished_count() > 0:
            events = dict(poller.poll())

            assert events.get(self.rep_socket) == zmq.POLLIN

            message = recv_pyobj_gz(self.rep_socket)

            # and respond
            logger.info(
                "[%s/%i] %s",
                str(self.done_count()).rjust(len(str(len(self.tstates))), "0"),
                len(self.tstates),
                message.get_summary(),
                )

            sender = self.wstates.get(message.sender)

            if sender is None:
                sender = WorkerState(message.sender)

                self.wstates[sender.condor_id] = sender

            if isinstance(message, ApplyMessage):
                # task request
                sender.disassociate()
                sender.set_assigned(self.next_task())

                send_pyobj_gz(self.rep_socket, sender.assigned.task)
            elif isinstance(message, DoneMessage):
                # task result
                finished = sender.assigned
                was_done = sender.set_done()

                assert finished.task.key == message.key

                working_ids = [ws.condor_id for ws in finished.working]

                if working_ids:
                    try:
                        cargo.condor_hold(working_ids)
                    except subprocess.CalledProcessError:
                        logger.warning("unable to hold %s", working_ids)

                selected = self.next_task()

                if selected is None:
                    send_pyobj_gz(self.rep_socket, None)
                else:
                    sender.set_assigned(selected)

                    send_pyobj_gz(self.rep_socket, selected.task)

                if not was_done:
                    self.handler(finished.task, message.result)
            elif isinstance(message, InterruptedMessage):
                # worker interruption
                sender.set_interruption()

                self.rep_socket.send(bytes())
            elif isinstance(message, ErrorMessage):
                # worker exception
                sender.set_error()

                self.rep_socket.send(bytes())
            else:
                assert False

    def next_task(self):
        """Select the next task on which to work."""

        tstate = min(self.tstates.itervalues(), key = TaskState.score)

        if tstate.done:
            return None
        else:
            return tstate

    def done_count(self):
        """Return the number of completed tasks."""

        return sum(1 for t in self.tstates.itervalues() if t.done)

    def unfinished_count(self):
        """Return the number of unfinished tasks."""

        return sum(1 for t in self.tstates.itervalues() if not t.done)

def distribute_labor(tasks, workers = 32, handler = lambda _, x: x):
    """Distribute computation to remote workers."""

    import zmq

    logger.info("distributing %i tasks to %i workers", len(tasks), workers)

    # prepare zeromq
    context = zmq.Context()
    rep_socket = context.socket(zmq.REP)
    rep_port = rep_socket.bind_to_random_port("tcp://*")

    logger.debug("listening on port %i", rep_port)

    # launch condor jobs
    cluster = cargo.submit_condor_workers(workers, "tcp://%s:%i" % (socket.getfqdn(), rep_port))

    try:
        # XXX workaround for bizarre pyzmq sigint behavior
        try:
            return Manager(tasks, handler, rep_socket).manage()
        except KeyboardInterrupt:
            raise
    finally:
        # clean up condor jobs
        cargo.condor_rm(cluster)

        logger.info("removed condor jobs")

        # clean up zeromq
        rep_socket.close()
        context.term()

        logger.info("terminated zeromq context")

def do_or_distribute(requests, workers, handler = lambda _, x: x):
    """Distribute or compute locally."""

    tasks = map(Task.from_request, requests)

    if workers > 0:
        return distribute_labor(tasks, workers, handler)
    else:
        while tasks:
            task = tasks.pop()
            result = task()

            handler(task, result)


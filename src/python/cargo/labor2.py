"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

from __future__ import absolute_import

import sys
import time
import uuid
import zlib
import socket
import random
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

class ApplyMessage(object):
    """A worker wants a unit of work."""

    def __init__(self, sender):
        self.sender = sender

class ErrorMessage(object):
    """An error occurred in a task."""

    def __init__(self, sender, key, description):
        self.sender = sender
        self.key = key
        self.description = description

class InterruptedMessage(object):
    """A worker was interrupted."""

    def __init__(self, sender, key):
        self.sender = sender
        self.key = key

class DoneMessage(object):
    """A task was completed."""

    def __init__(self, sender, key, result):
        self.sender = sender
        self.key = key
        self.result = result

class Task(object):
    """One unit of distributable work."""

    def __init__(self, call, args = [], kwargs = {}):
        self.call = call
        self.args = args
        self.kwargs = kwargs
        self.key = uuid.uuid4()

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
        """Score this task according to work urgency."""

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

        if self.assigned is not None:
            self.assigned.working.remove(self)

        self.assigned = tstate
        self.timestamp = time.time()

        self.assigned.working.add(self)

    def set_interruption(self):
        """Change worker state in response to interruption."""

        if self.assigned is not None:
            self.assigned.working.remove(self)

    def set_error(self):
        """Change worker state in response to error."""

        if self.assigned is not None:
            self.assigned.working.remove(self)

def distribute_labor_on(task_list, handler, rep_socket):
    import zmq

    # manage worker and task states
    tstates = dict((t.key, TaskState(t)) for t in task_list)
    wstates = {}

    def next_task():
        tstate = min(tstates.itervalues(), key = TaskState.score)

        if tstate.done:
            return None
        else:
            return tstate

    def done_count():
        count = 0

        for tstate in tstates.itervalues():
            if tstate.done:
                count += 1

        return count

    # receive messages
    poller = zmq.Poller()

    poller.register(rep_socket, zmq.POLLIN)

    while next_task() is not None:
        events = dict(poller.poll())

        if events.get(rep_socket) != zmq.POLLIN:
            continue

        message = recv_pyobj_gz(rep_socket)

        logger.debug("%s from %s", type(message), message.sender)

        # and respond to them
        if isinstance(message, ApplyMessage):
            wstate = wstates.get(message.sender)

            if wstate is None:
                wstate = WorkerState(message.sender)

                wstates[message.sender] = wstate

            wstate.set_assigned(next_task())

            send_pyobj_gz(rep_socket, wstate.assigned.task)
        elif isinstance(message, DoneMessage):
            wstate = wstates[message.sender]

            assert wstate.assigned.task.key == message.key

            done_tstate = wstate.assigned
            was_done = wstate.set_done()

            working_ids = [ws.condor_id for ws in done_tstate.working]

            if working_ids:
                cargo.condor_hold(working_ids)
                cargo.condor_release(working_ids)

            for other_wstate in list(done_tstate.working):
                other_wstate.set_done()

            next_tstate = next_task()

            if next_tstate is None:
                send_pyobj_gz(rep_socket, None)
            else:
                wstate.set_assigned(next_tstate)

                send_pyobj_gz(rep_socket, next_tstate.task)

            if not was_done:
                handler(done_tstate.task, message.result)
        elif isinstance(message, InterruptedMessage):
            wstates[message.sender].set_interruption()

            rep_socket.send(bytes())
        elif isinstance(message, ErrorMessage):
            logger.warning("worker %s hit error on %s:\n%s", message.sender, key, message.description)

            wstates[message.sender].set_error()

            rep_socket.send(bytes())
        else:
            logger.warning("unknown message type: %s", type(message))

        # be informative
        count = done_count()

        logger.info(
            "%i of %i tasks complete (%.2f%%)",
            count,
            len(tstates),
            count * 100.0 / len(tstates),
            )

def distribute_labor(tasks, workers = 32, handler = lambda _, x: x):
    """Distribute computation to remote workers."""

    import zmq

    logger.info("distributing %i tasks to %i workers", len(tasks), workers)

    # prepare zeromq
    context = zmq.Context()
    rep_socket = context.socket(zmq.REP)
    rep_port = rep_socket.bind_to_random_port("tcp://*")

    logger.info("listening on port %i", rep_port)

    # launch condor jobs
    cluster = cargo.submit_condor_workers(workers, "tcp://%s:%i" % (socket.getfqdn(), rep_port))

    try:
        return distribute_labor_on(tasks, handler, rep_socket)
    finally:
        # clean up condor jobs
        cargo.condor_rm(cluster)

        logger.info("cleaned up condor jobs")

        # clean up zeromq
        rep_socket.close()
        context.term()

        logger.info("cleaned up zeromq context")

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


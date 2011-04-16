"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import uuid
import zlib
import socket
import traceback
import cPickle as pickle
import cargo

logger = cargo.get_logger(__name__, level = "INFO")

class Task(object):
    """One unit of distributable work."""

    def __init__(self, call, args = [], kwargs = {}, extra = None):
        self.call = call
        self.args = args
        self.kwargs = kwargs
        self.extra = extra

    def __call__(self):
        return self.call(*self.args, **self.kwargs)

    def __iter__(self):
        yield self.call
        yield self.args
        yield self.kwargs
        yield self.extra

class ParentTask(Task):
    """One unit of distributable work."""

def task_from_request(request):
    """Build a task, if necessary."""

    if isinstance(request, Task):
        return request
    else:
        return Task(*request)

def send_pyobj_gz(socket, message):
    pickled = pickle.dumps(message)
    compressed = zlib.compress(pickled)

    socket.send(compressed)

def recv_pyobj_gz(socket):
    compressed = socket.recv()
    decompressed = zlib.decompress(compressed)
    unpickled = pickle.loads(decompressed)

    return unpickled

def distribute_labor_on(task_list, handler, rep_socket, pull_socket):
    import zmq

    # labor state
    unassigned = dict((uuid.uuid4(), task) for task in task_list)
    assigned = {}

    seen_count = len(task_list)
    done_count = 0

    # messaging preparation
    poller = zmq.Poller()

    poller.register(rep_socket, zmq.POLLIN)
    poller.register(pull_socket, zmq.POLLIN)

    # distribute work
    while assigned or unassigned:
        events = dict(poller.poll())

        # handle a request
        if rep_socket in events and events[rep_socket] == zmq.POLLIN:
            rep_socket.recv()

            # respond with an assignment
            if unassigned:
                (key, task) = unassigned.popitem()

                assigned[key] = task

                send_pyobj_gz(rep_socket, (key, task))

                logger.debug("handed out %s", key)
            else:
                send_pyobj_gz(rep_socket, None)

                logger.debug("handed out nothing")

        # handle an update
        if pull_socket in events and events[pull_socket] == zmq.POLLIN:
            (update, key, body) = recv_pyobj_gz(pull_socket)

            if update == "bailed":
                if key in assigned:
                    unassigned[key] = assigned.pop(key)

                logger.warning("worker bailed on task %s with:\n%s", key, body)
            elif update == "completed":
                if key in assigned:
                    task = assigned.pop(key)

                    done_count += 1

                    if isinstance(task, ParentTask):
                        for child in map(task_from_request, body):
                            unassigned[uuid.uuid4()] = child

                            seen_count += 1
                    else:
                        handler(task, body)

                    logger.debug("worker completed task %s", key)
                else:
                    logger.debug("worker completed task %i (dup)", key)
            else:
                logger.warning("unknown update type: %s", update)

        logger.info(
            "%i unassigned; %i assigned; %i complete (%.2f%%)",
            len(unassigned),
            len(assigned),
            done_count,
            done_count * 100.0 / seen_count,
            )

def distribute_labor(tasks, workers = 32, handler = lambda _, x: x):
    """Distribute computation to remote workers."""

    import zmq

    logger.info("distributing %i tasks to %i workers", len(tasks), workers)

    # prepare zeromq
    context = zmq.Context()
    rep_socket = context.socket(zmq.REP)
    rep_port = rep_socket.bind_to_random_port("tcp://*")
    pull_socket = context.socket(zmq.PULL)
    pull_port = pull_socket.bind_to_random_port("tcp://*")

    logger.info("listening on ports %i (rep) and %i (pull)", rep_port, pull_port)

    # launch condor jobs
    fqdn = socket.getfqdn()
    cluster = \
        cargo.submit_condor_workers(
            workers,
            "tcp://%s:%i" % (fqdn, rep_port),
            "tcp://%s:%i" % (fqdn, pull_port),
            )

    try:
        return distribute_labor_on(tasks, handler, rep_socket, pull_socket)
    finally:
        # clean up condor jobs
        cargo.condor_rm(cluster)

        logger.info("cleaned up condor jobs")

        # clean up zeromq
        rep_socket.close()
        pull_socket.close()
        context.term()

        logger.info("cleaned up zeromq context")

def do_or_distribute(requests, workers, handler = lambda _, x: x):
    """Distribute or compute locally."""

    tasks = map(task_from_request, requests)

    if workers > 0:
        return distribute_labor(tasks, workers, handler)
    else:
        while tasks:
            task = tasks.pop()
            result = task()

            if isinstance(task, ParentTask):
                for child in map(task_from_request, result):
                    tasks.append(child)
            else:
                handler(task, result)


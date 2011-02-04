"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import socket
import traceback
import zmq
import cargo

logger = cargo.get_logger(__name__, level = "INFO")

def distribute_labor_on(assignments, handler, rep_socket, pull_socket):
    # labor state
    ids = dict(enumerate(assignments))
    unassigned = set(ids)
    assigned = set()
    complete = {}

    # messaging preparation
    poller = zmq.Poller()

    poller.register(rep_socket, zmq.POLLIN)
    poller.register(pull_socket, zmq.POLLIN)

    # distribute work
    while unassigned or assigned:
        events = dict(poller.poll())

        # handle a request
        if rep_socket in events and events[rep_socket] == zmq.POLLIN:
            rep_socket.recv_pyobj()

            # respond with an assignment
            if unassigned:
                id_ = unassigned.pop()

                assigned.add(id_)

                rep_socket.send_pyobj((id_, assignments[id_]))

                logger.debug("handed out %s", id_)
            else:
                rep_socket.send_pyobj(None)

                logger.debug("handed out nothing")

        # handle an update
        if pull_socket in events and events[pull_socket] == zmq.POLLIN:
            (update, id_, body) = pull_socket.recv_pyobj()

            if update == "bailed":
                if id_ in assigned:
                    assigned.remove(id_)
                    unassigned.add(id_)

                logger.warning("worker bailed on assignment %i with:\n%s", id_, body)
            elif update == "completed":
                if id_ in assigned:
                    assigned.remove(id_)

                    complete[id_] = handler(body)

                    logger.debug("worker completed assignment %i", id_)
                else:
                    logger.debug("worker completed assignment %i (dup)", id_)
            else:
                logger.warning("unknown update type: %s", update)

        logger.info(
            "status: %i unassigned; %i assigned; %i complete",
            len(unassigned),
            len(assigned),
            len(complete),
            )

    return [x for (_, x) in sorted(complete.items())]

def distribute_labor(assignments, workers = 32, handler = lambda x: x):
    """
    Distribute computation to remote workers.
    """

    assignments = list(assignments)

    logger.info("distributing %i assignments to %i workers", len(assignments), workers)

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
        return distribute_labor_on(assignments, handler, rep_socket, pull_socket)
    finally:
        # clean up condor jobs
        cargo.condor_rm(cluster)

        logger.info("cleaned up condor jobs")

        # clean up zeromq
        rep_socket.close()
        pull_socket.close()
        context.term()

        logger.info("cleaned up zeromq context")

def distribute_or_labor(assignments, workers, handler = lambda x: x):
    """
    Distribute or compute locally.
    """

    if workers > 0:
        return distribute_labor(assignments, workers, handler)
    else:
        return [handler(call(*args, **kwargs)) for (call, args, kwargs) in assignments]


"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.work2 import main

    plac.call(main)

import numpy
import random
import traceback
import zmq
import cargo

logger = cargo.get_logger(__name__, level = "NOTSET")

def work_once(req_socket, task):
    """Request and/or complete a single unit of work."""

    # get an assignment
    if task is None:
        cargo.send_pyobj_gz(
            req_socket,
            cargo.labor2.ApplyMessage(condor_id),
            )

        task = cargo.recv_pyobj_gz(req_socket)

        if task is None:
            logger.info("received null assignment; terminating")

            return None

    # complete the assignment
    try:
        seed = abs(hash(task.key))

        logger.info("setting PRNG seed to %s", seed)

        numpy.random.seed(seed)
        random.seed(numpy.random.randint(2**32))

        logger.info("starting work on task %s", task.key)

        result = task()
    except KeyboardInterrupt, error:
        logger.warning("interruption during task %s", task.key)

        cargo.send_pyobj_gz(
            req_socket,
            cargo.labor2.InterruptedMessage(condor_id, task.key),
            )

        req_socket.recv()
    except BaseException, error:
        description = traceback.format_exc(error)

        logger.warning("error during task %s:\n%s", task.key, description)

        cargo.send_pyobj_gz(
            req_socket,
            cargo.labor2.ErrorMessage(condor_id, task.key, description),
            )

        req_socket.recv()
    else:
        logger.info("finished task %s", task.key)

        cargo.send_pyobj_gz(
            req_socket,
            cargo.labor2.DoneMessage(condor_id, task.key, result),
            )

        return cargo.recv_pyobj_gz(req_socket)

    return None

def work_loop(req_socket):
    """Repeatedly request and complete units of work."""

    task = None

    while True:
        try:
            task = work_once(req_socket, task)
        except Exception:
            raise

        if task is None:
            break

@plac.annotations(
    req_address = ("zeromq address of master"),
    condor_id = ("condor process specifier"),
    )
def main(req_address, condor_id):
    """Do arbitrary distributed work."""

    cargo.enable_default_logging()

    # connect to the work server
    logger.info("connecting to %s", req_address)

    context = zmq.Context()

    req_socket = context.socket(zmq.REQ)

    req_socket.connect(req_address) 

    # enter the work loop
    try:
        work_loop(req_socket)
    finally:
        logger.info("flushing sockets and terminating zeromq context")

        req_socket.close()
        context.term()

        logger.info("zeromq cleanup complete")


"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.work2 import main

    plac.call(main)

import time
import numpy
import random
import traceback
import zmq
import cargo

logger = cargo.get_logger(__name__, level = "NOTSET")

@plac.annotations(
    req_address = ("zeromq address of master"),
    condor_id = ("condor process specifier"),
    )
def main(req_address, condor_id):
    """Do arbitrary distributed work."""

    cargo.enable_default_logging()

    # connect to the work server
    context = zmq.Context()

    req_socket = context.socket(zmq.REQ)

    req_socket.connect(req_address) 

    # enter the work loop
    task = None

    try:
        while True:
            # get an assignment
            if task is None:
                cargo.send_pyobj_gz(
                    req_socket,
                    cargo.labor2.ApplyMessage(condor_id),
                    )

                task = cargo.recv_pyobj_gz(req_socket)

                if task is None:
                    logger.info("received null assignment; terminating")

                    break

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

                break
            except BaseException, error:
                description = traceback.format_exc(error)

                logger.warning("error during task %s:\n%s", task.key, description)

                cargo.send_pyobj_gz(
                    req_socket,
                    cargo.labor2.ErrorMessage(condor_id, task.key, description),
                    )
                req_socket.recv()

                break
            else:
                logger.info("finished task %s", task.key)

                cargo.send_pyobj_gz(
                    req_socket,
                    cargo.labor2.DoneMessage(condor_id, task.key, result),
                    )

                task = cargo.recv_pyobj_gz(req_socket)
    finally:
        logger.info("flushing sockets and terminating zeromq context")

        time.sleep(64) # XXX shouldn't be necessary

        req_socket.close()
        context.term()

        logger.info("zeromq cleanup complete")


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
    req_address = ("address for requests", "positional"),
    push_address = ("address for updates", "positional"),
    period = ("new-work poll period (s)", "option", None, float),
    )
def main(req_address, push_address, period = 8):
    """Do arbitrary distributed work."""

    cargo.enable_default_logging()

    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(req_address) 
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect(push_address) 

    try:
        while True:
            # get an assignment
            req_socket.send(bytes())

            response = cargo.recv_pyobj_gz(req_socket)

            if response is None:
                logger.info("received null assignment; sleeping for %i s", period)

                time.sleep(period)
            else:
                # complete the assignment
                (key, task) = response

                logger.info("starting task %s", key)

                numpy.random.seed(abs(hash(key)))
                random.seed(numpy.random.randint(2**32))

                try:
                    result = task()
                except BaseException, error:
                    description = traceback.format_exc(error)

                    logger.info("error! bailing on task %s with:\n%s", key, description)

                    cargo.send_pyobj_gz(push_socket, ("bailed", key, description))

                    time.sleep(4) # a gesture toward avoiding a failure stampede
                else:
                    logger.info("finished task %s", key)

                    cargo.send_pyobj_gz(push_socket, ("completed", key, result))
    finally:
        logger.info("flushing sockets and terminating zeromq context")

        time.sleep(64) # XXX

        req_socket.close()
        push_socket.close()
        context.term()

        logger.info("zeromq cleanup complete")


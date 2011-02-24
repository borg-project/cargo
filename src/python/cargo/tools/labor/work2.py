"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.work2 import main

    plac.call(main)

import time
import traceback
import zmq
import cargo

logger = cargo.get_logger(__name__, level = "NOTSET")

@plac.annotations(
    req_address = ("address for requests", "positional"),
    push_address = ("address for updates", "positional"),
    )
def main(req_address, push_address):
    """
    Do arbitrary distributed work.
    """

    cargo.enable_default_logging()

    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(req_address) 
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect(push_address) 

    try:
        while True:
            # get an assignment
            req_socket.send_pyobj(None)

            response = req_socket.recv_pyobj()

            if response is None:
                logger.info("received null assignment; terminating")

                break
            else:
                # complete the assignment
                (id_, assignment) = response
                work = assignment[0]
                work_args = assignment[1] if len(assignment) > 1 else []
                work_kwargs = assignment[2] if len(assignment) > 2 else {}

                logger.info("working on assignment %i", id_)

                try:
                    result = work(*work_args, **work_kwargs)
                except BaseException, error:
                    logger.info("error; bailing!")

                    push_socket.send_pyobj(("bailed", id_, traceback.format_exc(error)))

                    raise
                else:
                    logger.info("assignment %i complete", id_)

                    push_socket.send_pyobj(("completed", id_, result))
    finally:
        logger.info("flushing sockets and terminating zeromq context")

        time.sleep(64) # XXX

        req_socket.close()
        push_socket.close()
        context.term()

        logger.info("zeromq cleanup complete")


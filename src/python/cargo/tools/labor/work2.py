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

logger = cargo.get_logger(__name__)

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
                (id_, (work, work_args, work_kwargs)) = response

                try:
                    result = work(*work_args, **work_kwargs)
                except BaseException, error:
                    push_socket.send_pyobj(("bailed", id_, traceback.format_exc(error)))

                    raise
                else:
                    push_socket.send_pyobj(("completed", id_, result))
    finally:
        req_socket.close()
        push_socket.close()
        context.term()


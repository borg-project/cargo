"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.work2 import main

    plac.call(main)

import zmq
import cargo

logger = cargo.get_logger(__name__)

@plac.annotations(
    address = ("address of employer", "positional"),
    )
def main(address):
    """
    Do arbitrary distributed work.
    """

    cargo.enable_default_logging()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    socket.connect(address) 

    while True:
        # get an assignment
        socket.send_pyobj(("get", None, None))

        (work_id, (work, work_args, work_kwargs)) = socket.recv_pyobj()

        # complete the assignment
        result = work(*work_args, **work_kwargs)

        # return the result
        socket.send_pyobj(("give", work_id, result))
        socket.recv_pyobj()


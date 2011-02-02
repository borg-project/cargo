"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import zmq
import cargo

logger = cargo.get_logger(__name__, level = "INFO")

def distribute_labor_on(assignments, socket):
    unfinished = range(len(assignments))
    results = [None] * len(assignments)
    by_id = dict(enumerate(assignments))
    next_ = 0

    while unfinished:
        (action, id_, result) = socket.recv_pyobj()

        if action == "get":
            id_ = unfinished[next_]
            next_ += 1

            socket.send_pyobj((id_, by_id[id_]))

            logger.info("handed out %s (%i)", id_, next_)
        elif action == "give":
            if id_ in unfinished:
                done = unfinished.index(id_)

                if done < next_:
                    next_ -= 1

                del unfinished[done]
                del by_id[id_]

            socket.send_pyobj(None)

            results[id_] = result

            logger.info("assignment %s done; %i left", id_, len(unfinished))
        else:
            log.warning("unknown action")

        if next_ >= len(unfinished):
            next_ = 0

def distribute_labor(assignments):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://*")

    logger.info("listening on port %i", port)

    try:
        return distribute_labor_on(assignments, socket)
    finally:
        socket.close()
        context.term()


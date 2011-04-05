"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

__all__ = [
    "call_external",
    ]

import Queue as queue
import multiprocessing
import cargo

class CallProcess(multiprocessing.Process):
    def __init__(self, method, to_master):
        self._method = method
        self._to_master = to_master

    def run(self):
        result = self._method()
        rutime = resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self._to_master.put((result, rutime))

class TrackProcess(multiprocessing.Process):
    def __init__(self, tracked, limit, to_master):
        self._tracked = tracked_pid
        self._limit = limit
        self._to_master = to_master

    def run(self):
        while cargo.get_pid_utime(self._tracked) < self._limit:
            time.sleep(1.0)

        rutime = resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self._to_master.put((result, rutime))

def call_external(method, cpu_seconds = None):
    queue = None
    call_child = None
    track_child = None

    try:
        queue = multiprocessing.Queue()
        call_child = ExternalCall(method, queue)

        child.start()

        track_child = ExternalCall(call_child.pid, queue)

        track_child.start()

        try:
            return queue.get(timeout = timeout)
        except queue.Empty:
            return None
    finally:
        if queue is not None:
            queue.close()
            queue.join_thread()
        if child is not None and child.is_alive():
            child.terminate()
            child.join()


"""
cargo/unix/accounting.py

Track resource usage.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os
import re
import time
import errno
import select
import signal
import subprocess

from datetime import timedelta
from itertools import count
from cargo.log import get_logger
from cargo.errors import (
    print_ignored_error,
    Raised,
    )
from cargo.unix.proc import ProcessStat
from cargo.unix.sessions import (
    kill_session,
    spawn_pty_session,
    )

log = get_logger(__name__)

class SessionTimeAccountant(object):
    """
    Track the total CPU (user) time for members of a session.

    Process accounting under Linux is a giant pain. In the general case, it is
    literally impossible (without patching the kernel or some such craziness).
    Whatever. We do our best. Slightly fancier schemes are available, but
    they're not much fancier---they're mostly good only at making it harder for
    processes to actively evade being charged. For primarily long-running
    processes that act in good faith, we should do ok.
    """

    def __init__(self, sid):
        """
        Initialize.
        """

        self.sid = sid
        self.charged = {}

    def audit(self):
        """
        Update estimates.
        """

        for p in ProcessStat.in_session(self.sid):
            self.charged[p.pid] = p.user_time

    def get_total(self, reaudit = True):
        """
        Return the total recorded CPU time.
        """

        if reaudit:
            self.audit()

        return sum(self.charged.itervalues(), timedelta())

class PollingReader(object):
    """
    Read from file descriptors with timeout.
    """

    def __init__(self, fd):
        """
        Initialize.
        """

        self.fd = fd
        self.polling = select.poll()

        self.polling.register(fd, select.POLLIN)

    def read(self, timeout = -1):
        """
        Read with an optional timeout.
        """

        changed = self.polling.poll(timeout * 1000)

        for (fd, event) in changed:
            if event & select.POLLIN:
                # POLLHUP is level-triggered; we'll be back if it was missed
                return os.read(fd, 65536)
            elif event & select.POLLHUP:
                return ""

        return None

def run_cpu_limited(arguments, limit, resolution = 0.25):
    """
    Spawn a subprocess whose process tree is granted limited CPU (user) time.

    The subprocess must not expect input. This method is best suited to
    processes which may run for a reasonable amount of time (eg, at least
    several seconds); it will be fairly inefficient (and ineffective) at
    fine-grained limiting of CPU allocation to short-duration processes.

    We run the process in a pseudo-terminal and read its output. Every time we
    receive a chunk of data, or every C{resolution} seconds, we estimate the
    total CPU time used by the session---and store that information with the
    chunk of output, if any. After at least C{limit} CPU seconds have been used
    by the spawned session, or after the session leader terminates, whichever
    is first, the session is killed, the session leader waited on, and any data
    remaining in the pipe is read.

    Final elapsed CPU time for process trees which do not exceed their CPU time
    limit is taken from kernel-reported resource usage, which includes the sum
    of all directly and indirectly waited-on children. It will be accurate in
    the common case where processes correctly wait on their children, and
    inaccurate in cases where zombies are reparented to init. Elapsed CPU time
    for process trees which exceed their CPU time limit is taken from the /proc
    accounting mechanism used to do CPU time limiting, and will always be at
    least the specified limit.
    """

    # sanity
    if not arguments:
        raise ArgumentError()

    # various constants
    KILL_DELAY_SECONDS = 0.1
    WAITS_BEFORE_SIG9 = 64
    WAITS_AFTER_SIG9 = 16

    # start the run
    child_fd = None
    child_pid = None

    try:
        # start running the child process
        (child_pid, child_fd) = spawn_pty_session(arguments)

        # read the child's output while accounting (note that the session id
        # is, under Linux, the pid of the session leader)
        chunks = []
        accountant = SessionTimeAccountant(child_pid)
        reader = PollingReader(child_fd)

        while True:
            # read from and audit the child process
            chunk = reader.read(resolution)
            cpu_total = accountant.get_total()

            if chunk is not None:
                if chunk != "":
                    log.debug("got %i bytes at %f cpu seconds", len(chunk), cpu_total)

                    chunks.append((cpu_total, chunk))
                else:
                    break

            if cpu_total >= limit:
                break

        # make sure that the session is terminated; we haven't yet waited on our
        # child process, so no new session should yet have the same session id;
        # we first kill the direct child so that it has a chance to do cleanup
        os.kill(child_pid, signal.SIGTERM)

        for i in xrange(16):
            session_size = len(list(ProcessStat.in_session(child_pid)))

            if session_size == 1:
                break
            elif i < 15:
                time.sleep(KILL_DELAY_SECONDS)

        # then nuke any grandchildren in the session; these will reparent
        if session_size > 1:
            log.note("%i orphan(s) survived; nuking the session from orbit", session_size - 1)

            kill_session(child_pid, signal.SIGKILL)

        # finally, wait on the direct child
        for i in xrange(WAITS_BEFORE_SIG9 + WAITS_AFTER_SIG9):
            (exit_pid, termination, usage) = os.wait4(child_pid, os.WNOHANG)

            if exit_pid == 0:
                # the process has not yet terminated
                if i == WAITS_BEFORE_SIG9:
                    # nuke it (again, in some cases)
                    os.kill(child_pid, signal.SIGKILL)

                time.sleep(KILL_DELAY_SECONDS)
            else:
                # the process is dead
                child_pid = None

                break
    except:
        # something has gone awry, so we need to kill our children
        raised = Raised()

        if child_pid is not None:
            try:
                # nuke the entire session
                kill_session(child_pid, signal.SIGKILL)

                # briefly wait for the child to avoid leaking it
                for i in xrange(WAITS_AFTER_SIG9):
                    try:
                        os.waitpid(child_pid, os.WNOHANG)
                    except OSError, error:
                        if error.errno != errno.ECHILD:
                            raise

                    time.sleep(KILL_DELAY_SECONDS)
            except:
                print_ignored_error()

        raised.re_raise()
    else:
        # grab any output left in the pipe's kernel buffer
        chunk = reader.read()

        if chunk:
            chunks.append((cpu_total, chunk))

        # unpack the exit status
        if os.WIFEXITED(termination):
            exit_status = os.WEXITSTATUS(termination)
            elapsed = timedelta(seconds = usage.ru_utime)
            gap = abs(elapsed - cpu_total)

            if gap > timedelta(seconds = 1):
                log.warning("gap of %s between rusage and /proc reporting", gap)
        else:
            exit_status = None
            elapsed = cpu_total

        # done
        return (chunks, elapsed, exit_status)
    finally:
        # let's not leak file descriptors
        if child_fd is not None:
            os.close(child_fd)


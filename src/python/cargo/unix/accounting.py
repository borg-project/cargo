"""
cargo/unix/accounting.py

Track resource usage.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.unix.accounting import main

    raise SystemExit(main())

import os
import re
import sys
import time
import errno
import select
import signal

from datetime import timedelta
from itertools import count
from cargo.log import get_logger
from cargo.errors import Raised
from cargo.unix.proc import ProcessStat
from cargo.unix.sessions import (
    kill_session,
    spawn_session,
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

    def __init__(self, fds):
        """
        Initialize.
        """

        self.fds     = fds
        self.polling = select.poll()

        for fd in fds:
            self.polling.register(fd, select.POLLIN)

    def unregister(self, fds):
        """
        Unregister descriptors.
        """

        for fd in fds:
            self.polling.unregister(fd)
            self.fds.remove(fd)

    def read(self, timeout = -1):
        """
        Read with an optional timeout.
        """

        changed = self.polling.poll(timeout * 1000)

        for (fd, event) in changed:
            log.debug("event on fd %i is %#o", fd, event)

            if event & select.POLLIN:
                # POLLHUP is level-triggered; we'll be back if it was missed
                return (fd, os.read(fd, 65536))
            elif event & select.POLLHUP:
                return (fd, "")
            else:
                raise IOError("unexpected poll response %#o from file descriptor" % event)

        return (None, None)

def run_cpu_limited(
    arguments,
    limit,
    pty         = False,
    environment = {},
    resolution  = 0.5,
    ):
    """
    Spawn a subprocess whose process tree is granted limited CPU (user) time.

    @param environments Override specific existing environment variables.

    The subprocess must not expect input. This method is best suited to
    processes which may run for a reasonable amount of time (eg, at least
    several seconds); it will be fairly inefficient (and ineffective) at
    fine-grained limiting of CPU allocation to short-duration processes.

    We run the process and read its output. Every time we receive a chunk of
    data, or every C{resolution} seconds, we estimate the total CPU time used
    by the session---and store that information with the chunk of output, if
    any. After at least C{limit} CPU seconds have been used by the spawned
    session, or after the session leader terminates, whichever is first, the
    session is killed, the session leader waited on, and any data remaining in
    the pipe is read.

    If C{pty} is specified, the process is run in a pseudo-terminal, stderr is
    merged with stdout, and the process is unlikely to buffer its output.

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
    WAITS_BEFORE_SIG9  = 64
    WAITS_AFTER_SIG9   = 16

    # start the run
    child_fd  = None
    child_pid = None
    censored  = False

    log.debug("running %s for %s", arguments, limit)

    try:
        # start running the child process
        if pty:
            (child_pid, child_fd) = spawn_pty_session(arguments, environment)
            child_fds             = [child_fd]
            out_chunks            = []
            err_chunks            = None
            fd_chunks             = {child_fd: out_chunks}
        else:
            popened      = spawn_session(arguments, environment)
            child_pid    = popened.pid
            out_chunks   = []
            err_chunks   = []
            fd_chunks    = {
                popened.stdout.fileno(): out_chunks,
                popened.stderr.fileno(): err_chunks,
                }
            child_fds    = fd_chunks.keys()

        log.debug("spawned child with pid %i", child_pid)

        # read the child's output while accounting (note that the session id
        # is, under Linux, the pid of the session leader)
        chunks     = []
        accountant = SessionTimeAccountant(child_pid)
        reader     = PollingReader(child_fds)

        while reader.fds:
            # read from and audit the child process
            (chunk_fd, chunk) = reader.read(resolution)
            cpu_total         = accountant.get_total()

            if chunk is not None:
                chunks = fd_chunks[chunk_fd]

                if chunk != "":
                    log.detail("got %i bytes at %s (user time)", len(chunk), cpu_total)
                    log.debug("chunk follows:\n%s", chunk)

                    chunks.append((cpu_total, chunk))
                else:
                    log.debug("fd %i closed; assuming child terminated", chunk_fd)

                    reader.unregister([chunk_fd])

            if cpu_total >= limit:
                log.debug("exceeded user time limit")

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

        # then nuke any process(es) left in the session; grandchildren will reparent
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

        log.warning("something went awry! (our pid is %i)", os.getpid())

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
                Raised().print_ignored()

        raised.re_raise()
    else:
        # grab any output left in the kernel buffers
        while reader.fds:
            (chunk_fd, chunk) = reader.read()

            if chunk:
                fd_chunks[chunk_fd].append((cpu_total, chunk))
            else:
                reader.unregister(chunk_fd)

        # unpack the exit status
        if os.WIFEXITED(termination):
            exit_status = os.WEXITSTATUS(termination)
        else:
            exit_status = None

        # done
        log.debug("completed (status %s); returning %i chunks", exit_status, len(chunks))

        # FIXME return a namedtuple?
        return (
            out_chunks,
            err_chunks,
            timedelta(seconds = usage.ru_utime),
            cpu_total,
            exit_status,
            )
    finally:
        # let's not leak file descriptors
        if child_fd is not None:
            os.close(child_fd)

def main():
    """
    Run a CPU-limited subprocess for testing.
    """

    (out_chunks, err_chunks, usage_elapsed, proc_elapsed, exit_status) = \
        run_cpu_limited(
            sys.argv[2:],
            timedelta(seconds = float(sys.argv[1])),
            pty = False,
            )

    print "usage_elapsed:", usage_elapsed
    print "proc_elapsed:", proc_elapsed
    print "output chunks follow"

    for (time, chunk) in out_chunks:
        print "================ (out)", time
        print chunk

    for (time, chunk) in err_chunks:
        print "================ (out)", time
        print chunk


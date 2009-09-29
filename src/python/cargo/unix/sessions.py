"""
cargo/unix/sessions.py

Deal with (pseudo) terminal sessions.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os
import sys
import pty

from cargo.log import get_logger
from cargo.unix.proc import ProcessStat
from cargo.errors import print_ignored_error

log = get_logger(__name__)

def spawn_pty_session(arguments):
    """
    Fork a subprocess in a pseudo-terminal.
    """

    # FIXME race condition: might leak the child if we're signaled

    dup_stderr_fd = os.dup(sys.stderr.fileno())
    (child_pid, child_fd) = pty.fork()

    if child_pid == 0:
        # we are the child; use the parent's stderr instead of merging to stdout
        os.dup2(dup_stderr_fd, sys.stderr.fileno())

        # then exec the command
        try:
            os.execvp(arguments[0], arguments)
        except:
            print_ignored_error()

        # something went wrong
        os._exit(os.EX_OSERR)
    else:
        # we are the parent
        os.close(dup_stderr_fd)

    return (child_pid, child_fd)

def kill_session(sid, number):
    """
    Send signal C{number} to all processes in session C{sid}.

    Imperfect: see the warnings associated with ProcessStat.

    @return: The number of processes signaled.
    """

    nkilled = 0

    for process in ProcessStat.in_session(sid):
        os.kill(process.pid, number)

        nkilled += 1

    return nkilled


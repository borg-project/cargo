"""
cargo/unix/sessions.py

Deal with (pseudo) terminal sessions.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from cargo.unix.sessions import main

    raise SystemExit(main())

import os
import pty
import sys
import subprocess

from os              import (
    putenv,
    fdopen,
    )
from functools       import partial
from subprocess      import (
    Popen,
    call,
    )
from cargo.log       import get_logger
from cargo.unix.proc import ProcessStat
from cargo.errors    import Raised

log = get_logger(__name__)

def _child_preexec(environment):
    """
    Run in the child code prior to execution.
    """

    # update the environment
    for (key, value) in environment.iteritems():
        putenv(key, str(value))

    # start our own session
    os.setsid()

def spawn_pipe_session(arguments, environment = {}):
    """
    Spawn a subprocess in its own session.

    @see: spawn_pty_session
    """

    # launch the subprocess
    popened = \
        Popen(
            arguments,
            close_fds  = True,
            stdin      = subprocess.PIPE,
            stdout     = subprocess.PIPE,
            stderr     = subprocess.PIPE,
            preexec_fn = partial(_child_preexec, environment),
            )

    popened.stdin.close()

    return popened

def spawn_pty_session(arguments, environment = {}):
    """
    Spawn a subprocess in its own session, with stdout routed through a pty.

    @see: spawn_pty_session
    """

    # build a pty
    (master_fd, slave_fd) = pty.openpty()

    log.debug("opened pty %s", os.ttyname(slave_fd))

    try:
        # launch the subprocess
        popened        = \
            Popen(
                arguments,
                close_fds  = True,
                stdin      = slave_fd,
                stdout     = slave_fd,
                stderr     = subprocess.PIPE,
                preexec_fn = partial(_child_preexec, environment),
                )
        popened.stdout = os.fdopen(master_fd)

        os.close(slave_fd)

        return popened
    except:
        raised = Raised()

        try:
            if master_fd is not None:
                os.close(master_fd)
            if slave_fd is not None:
                os.close(slave_fd)
        except:
            Raised().print_ignored()

        raised.re_raise()

def kill_session(sid, number):
    """
    Send signal C{number} to all processes in session C{sid}.

    Theoretically imperfect, but should be consistently effective---almost
    certainly paranoid overkill---in practice.

    @return: The number of processes signaled.
    """

    # why do we pkill multiple times? because we're crazy.
    for i in xrange(2):
        exit_code = call(["pkill", "-%i" % number, "-s", "%i" % sid])

        if exit_code not in (0, 1):
            raise RuntimeError("pkill failure")


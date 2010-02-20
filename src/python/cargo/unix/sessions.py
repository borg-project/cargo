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
import fcntl

from os import (
    pipe,
    putenv,
    execvp,
    )
from cPickle import (
    dumps,
    loads,
    )
from traceback import format_exception
from cargo.log import get_logger
from cargo.unix.proc import ProcessStat
from cargo.errors import Raised

log = get_logger(__name__)

def spawn_pty_session(arguments, environment = {}):
    """
    Fork a subprocess in a pseudo-terminal.

    Merges child stderr to stdout; waits for successful exec.

    @param environment: Additional environment variables; putenved.
    """

    # FIXME race condition: might leak the child if we're signaled

    (error_read_fd, error_write_fd) = pipe()
    (child_pid, child_fd)           = pty.fork()

    if child_pid == 0:
        # close the error pipe on execvp success
        try:
            old_flags = fcntl.fcntl(error_write_fd, fcntl.F_GETFD)

            fcntl.fcntl(error_write_fd, fcntl.F_SETFD, old_flags | fcntl.FD_CLOEXEC)
        except:
            Raised().print_ignored()

        # then exec the command
        try:
            # set our environment
            for (key, value) in environment.iteritems():
                putenv(key, str(value))

            execvp(arguments[0], arguments)
        except:
            # received an exception; try to pass it to the parent
            raised = Raised()

            try:
                # the child_traceback attribute is also used by subprocess.Popen
                raised.value.child_traceback = ''.join(raised.format())

                os.write(error_write_fd, dumps(raised.value))
            except:
                raised.print_ignored()
                Raised().print_ignored()

        # something went wrong
        os.close(error_write_fd)
        os._exit(os.EX_OSERR)
    else:
        # we are the parent; wait for child exec success or failure
        os.close(error_write_fd)

        error_data = os.read(error_read_fd, 2**20)

        os.close(error_read_fd)

        if error_data:
            os.waitpid(child_pid, 0)

            child_error = loads(error_data)

            raise child_error

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

def main():
    """
    Run a child under a pty.
    """

    (child_pid, child_fd) = spawn_pty_session(sys.argv[1:])

    while True:
        data = os.read(child_fd, 1024)

        if data is None:
            break
        else:
            print data,


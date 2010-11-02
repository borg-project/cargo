"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                  import call
    from cargo.tools.triggered import main

    call(main)

import pyinotify

from plac      import annotations
from pyinotify import ProcessEvent
from cargo.log import get_logger

logger = get_logger(__name__, level = "NOTSET")

class TriggerHandler(ProcessEvent):
    """
    Handle relevant filesystem events.
    """

    def process_default(self, event):
        """
        Handle a filesystem event.
        """

        logger.info("filesystem event: %s", event)

def execute_command(command, tmux):
    """
    Execute the triggered command.
    """

    from subprocess import call

    logger.info("executing triggered command")

    if tmux:
        #call(["tmux", "-q", "setw", "window-status-fg", "green"])
        call(["tmux", "-q", "display-message", "build active"])

    call(command)

    if tmux:
        #call(["tmux", "-q", "setw", "window-status-fg", "default"])
        call(["tmux", "-q", "display-message", "build complete"])

@annotations(
    path       = ("path to watch"                               ),
    executable = ("triggered command"                           ),
    arguments  = ("command arguments"                           ),
    no_tmux    = ("disable tmux interaction", "option"          ),
    timeout    = ("wait; coalesce events"   , "option", "t", int),
    )
def main(path, executable, no_tmux = False, timeout = 250, *arguments):
    """
    Run something in response to changes in a directory.
    """

    # enable logging
    from cargo.log import enable_default_logging

    enable_default_logging()

    # prepare the notification framework
    from pyinotify import (
        Notifier,
        WatchManager,
        )

    command  = [executable] + list(arguments)
    manager  = WatchManager()
    handler  = TriggerHandler()
    notifier = Notifier(manager, handler)

    manager.add_watch(
        path,
        pyinotify.IN_CREATE | pyinotify.IN_DELETE | pyinotify.IN_MODIFY,
        rec = True,
        )

    # watch for and respond to events
    while True:
        triggered = notifier.check_events()

        if triggered:
            # coalesce events
            notifier.read_events()
            notifier.process_events()

            if timeout is not None:
                while notifier.check_events(timeout = timeout):
                    notifier.read_events()
                    notifier.process_events()

            # run the command
            execute_command(command, not no_tmux)


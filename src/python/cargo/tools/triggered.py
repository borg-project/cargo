"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                  import call
    from cargo.tools.triggered import main

    call(main)

# XXX can we modify screen's status bar?

import pyinotify

from plac      import annotations
from pyinotify import ProcessEvent

class TriggerHandler(ProcessEvent):
    """
    Handle relevant filesystem events.
    """

    def my_init(self, command, tmux):
        """
        Initialize.
        """

        self._command = command
        self._tmux    = tmux

    def process_default(self, event):
        """
        Handle a filesystem event.
        """

        from subprocess import call

        if self._tmux:
            #call(["tmux", "-q", "setw", "window-status-fg", "green"])
            call(["tmux", "-q", "display-message", "build active"])

        call(self._command)

        if self._tmux:
            #call(["tmux", "-q", "setw", "window-status-fg", "default"])
            call(["tmux", "-q", "display-message", "build complete"])

@annotations(
    path       = ("path to watch"                         ),
    executable = ("triggered command"                     ),
    arguments  = ("command arguments"                     ),
    tmux       = ("enable tmux interaction", "option", "t"),
    )
def main(path, executable, tmux = True, *arguments):
    """
    Run something in response to changes in a directory.
    """

    from pyinotify import (
        Notifier,
        WatchManager,
        )

    command  = [executable] + list(arguments)
    manager  = WatchManager()
    handler  = TriggerHandler(command = command, tmux = tmux)
    notifier = Notifier(manager, handler)

    manager.add_watch(
        path,
        pyinotify.IN_CREATE | pyinotify.IN_DELETE | pyinotify.IN_MODIFY,
        rec = True,
        )

    notifier.loop()


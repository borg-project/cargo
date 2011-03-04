"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from cargo.tools.triggered import main

    plac.call(main)

import subprocess
import pyinotify
import cargo

logger = cargo.get_logger(__name__, level = "NOTSET")

class TriggerHandler(pyinotify.ProcessEvent):
    """
    Handle relevant filesystem events.
    """

    def process_default(self, event):
        """
        Handle a filesystem event.
        """

        logger.info("filesystem event: %s", event)

def execute_command(command, tmux, window):
    """
    Execute the triggered command.
    """

    logger.info("executing triggered command")

    if tmux:
        if window is None:
            subprocess.call(["tmux", "-q", "display-message", "command active"])
        else:
            subprocess.call(["tmux", "-q", "setw", "-t", str(window), "window-status-bg", "green"])

    status = subprocess.call(command)

    if tmux:
        if window is None:
            if status == 0:
                message = "command successful"
            else:
                message = "command failed"

            subprocess.call(["tmux", "-q", "display-message", message])
        else:
            if status == 0:
                color = "default"
            else:
                color = "red"

            subprocess.call(["tmux", "-q", "setw", "-t", str(window), "window-status-bg", color])

@plac.annotations(
    path = ("path to watch"),
    executable = ("triggered command"),
    arguments = ("command arguments"),
    no_tmux = ("disable tmux interaction", "option"),
    window = ("tmux window number", "option"),
    timeout = ("wait; coalesce events", "option", "t", int),
    )
def main(path, executable, no_tmux = False, timeout = 250, window = None, *arguments):
    """
    Run something in response to changes in a directory.
    """

    # enable logging
    cargo.enable_default_logging()

    # prepare the notification framework
    command  = [executable] + list(arguments)
    manager  = pyinotify.WatchManager()
    handler  = TriggerHandler()
    notifier = pyinotify.Notifier(manager, handler)

    manager.add_watch(
        path,
        pyinotify.IN_CREATE | pyinotify.IN_DELETE | pyinotify.IN_MODIFY,
        rec = True,
        )

    # watch for and respond to events
    try:
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
                execute_command(command, not no_tmux, window)
    finally:
        if not no_tmux:
            subprocess.call(["tmux", "-q", "setw", "-t", str(window), "window-status-bg", "default"])


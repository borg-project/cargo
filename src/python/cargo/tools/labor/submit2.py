"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import plac

if __name__ == "__main__":
    from cargo.tools.labor.submit2 import main

    plac.call(main)

import cargo

logger = cargo.get_logger(__name__, level = "NOTE")

@plac.annotations(
    workers = ("number of workers to submit", "positional", None, int),
    req_address = ("address for requests", "positional"),
    push_address = ("address for updates", "positional"),
    matching = ("node match restriction", "option"),
    condor_home = ("job submission directory", "option"),
    )
def main(
    workers,
    req_address,
    push_address,
    matching = cargo.defaults.condor_matching,
    condor_home = cargo.default_condor_home(),
    ):
    """
    Submit workers to condor.
    """

    cargo.enable_default_logging()

    submit_workers(
        workers,
        req_address,
        push_address,
        matching,
        condor_home = condor_home,
        )


"""
cargo/condor/host.py

Host individual condor jobs.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os
import sys
import cPickle as pickle

def main():
    """
    Application entry point.
    """

    # make the job identifier obvious
    process_number  = int(os.environ["CONDOR_PROCESS"])
    cluster_number  = int(os.environ["CONDOR_CLUSTER"])
    identifier_path = "JOB_IS_%i.%i" % (cluster_number, process_number)

    open(identifier_path, "w").close()

    # load and run the job
    job = pickle.load(sys.stdin)

    job.run()

if __name__ == "__main__":
    main()

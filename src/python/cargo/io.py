"""
cargo/io.py

Operations associated with the filesystem.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os.path
import fnmatch
import mimetypes
import subprocess

from bz2 import BZ2File
from gzip import GzipFile

def files_under(path, pattern = "*"):
    """
    Iterate over the set of paths to files in the specified directory tree.

    @param path: Specified path.
    @param pattern: Optional filter pattern.
    """

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                yield os.path.join(dirpath, filename)

def bq(arguments, cwd = None):
    """
    Spawn a process and return its output.
    """

    p = subprocess.Popen(arguments, cwd = cwd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    output = p.communicate()[0]

    return (output, p.returncode)

def openz(path, mode = 'r'):
    """
    Open a file, transparently [un]compressing it if a known compression extension is present.

    Uses the system MIME type database to detect the presence and type of
    compression encoding, if any. This method is not perfect, but should
    correctly handle common cases.
    """

    (mime_type, encoding) = mimetypes.guess_type(path)

    if encoding == "bzip2":
        return BZ2File(path, mode)
    elif encoding == "gzip":
        return GzipFile(path, mode)
    elif encoding is None:
        return open(path, mode)
    else:
        raise RuntimeError("unsupported file encoding")

def write_from_file(tf, ff, chunk_size = 2**16):
    """
    Write the contents of file object C{ff} to file object C{tf}.
    """

    while True:
        chunk = ff.read(chunk_size)

        if chunk:
            tf.write(chunk)
        else:
            return


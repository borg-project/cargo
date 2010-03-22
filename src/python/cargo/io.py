"""
cargo/io.py

Operations associated with the filesystem.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import os
import os.path
import errno
import hashlib
import threading
import mimetypes
import subprocess

from os.path      import (
    join,
    expanduser,
    expandvars,
    )
from bz2          import BZ2File
from gzip         import GzipFile
from fnmatch      import fnmatch
from cargo.errors import Raised

def expandpath(path, relative = ""):
    """
    Expand and (possibly) extend the specified path.
    """

    return join(relative, expandvars(expanduser(path)))

def files_under(path, pattern = "*"):
    """
    Iterate over the set of paths to files in the specified directory tree.

    @param path: Specified path.
    @param pattern: Optional filter pattern(s).
    """

    if isinstance(pattern, str):
        pattern = (str,)

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if any(fnmatch(filename, p) for p in pattern):
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
    Open a file, transparently [de]compressing it if a known compression extension is present.

    Uses the system MIME type database to detect the presence and type of
    compression encoding, if any. This method is not perfect, but should
    correctly handle common cases.
    """

    # bloody hack
    if ".xz" not in mimetypes.encodings_map:
        mimetypes.encodings_map[".xz"] = "xz"

    (mime_type, encoding) = mimetypes.guess_type(path)

    if encoding == "bzip2":
        return BZ2File(path, mode)
    elif encoding == "gzip":
        return GzipFile(path, mode)
    elif encoding == "xz":
        raise NotImplementedError()
    elif encoding is None:
        return open(path, mode)
    else:
        raise RuntimeError("unsupported file encoding")

def hash_file(path, algorithm_name = None):
    """
    Return a deterministic hash of the contents of C{bytes}.

    @return (algorithm_name, hash)
    """

    with open(path) as path_file:
        return hash_bytes(path_file.read(), algorithm_name)

def hash_bytes(bytes, algorithm_name = None):
    """
    Return a deterministic hash of C{bytes}.

    @return (algorithm_name, hash)
    """

    return hash_yielded_bytes([bytes])

def hash_yielded_bytes(iterator, algorithm_name = None):
    """
    Return a deterministic hash of the C{iterator} of byte strings.

    @return (algorithm_name, hash)
    """

    if algorithm_name is None:
        algorithm      = hashlib.sha512()
        algorithm_name = "sha512"
    else:
        algorithm = hashlib.new(algorithm_name)

    for bytes in iterator:
        algorithm.update(bytes)

    return (algorithm_name, algorithm.digest())

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

def write_file_atomically(path, data):
    """
    Write a temporary file, fsync, rename, and clean up.
    """

    # FIXME untested, unfinished, unused

    # grab the existing mode bits, if any
    try:
        stat_info = os.stat(path)
    except OSError, error:
        if error.errno != errno.ENOENT:
            raise
        else:
            stat_info = None

    # generate a unique temporary path
    (path_dir, path_base) = os.path.split(path)
    temp_path = os.path.join(path_dir, ".%s~%i" % (path_base, uuid4()))

    # write data to the temporary
    temp_fd = None

    try:
        temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat_info.st_mode)
        written = 0

        while written < len(data):
            written += os.write(temp_fd, data[written:])

        os.fdatasync(temp_fd)
        temp_fd_ = temp_fd
        temp_fd = None
        os.close(temp_fd_)
        os.rename(temp_path, path)
        os.unlink(temp_path)
    except:
        try:
            if temp_fd != None:
                os.close(temp_fd)

            os.unlink(temp_path)
        except:
            Raised().print_ignored()

def escape_for_latex(text):
    """
    Escape a text string for use in a LaTeX document.
    """

    return \
        replace_all(
            text,
            ("%", r"\%"),
            ("_", r"\_"))


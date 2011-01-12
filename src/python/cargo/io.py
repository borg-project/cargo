"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

__all__ = [
    "expandvars",
    "expandpath",
    "files_under",
    "uncompressed",
    "mkdtemp_scoped",
    "decompress_if",
    ]

import os
import os.path
import pwd
import errno
import shutil
import hashlib
import tempfile
import threading
import mimetypes
import subprocess
import cargo

from os           import (
    fsync,
    rename,
    )
from os.path      import (
    join,
    exists,
    expanduser,
    expandvars,
    )
from uuid         import (
    uuid4,
    uuid5,
    )
from shutil       import (
    copy2,
    )
from tempfile     import (
    gettempdir,
    )
from subprocess   import (
    Popen,
    check_call,
    )
from contextlib   import contextmanager
from cargo.errors import Raised

def write_named_tuples_to_csv(path, tuples):
    """
    Write a list of named tuples to a CSV file.
    """

    from csv import writer as csv_writer

    with open(path, "w") as file_:
        writer = csv_writer(file_)

        if tuples:
            names = tuples[0]._fields
        else:
            return

        writer.writerow(names)
        writer.writerows(tuples)

def cache_file(path, cache_dir = None, namespace = uuid4()):
    """
    Safely cache a file in some location.

    This function computes a path-unique name. If a file with that name already
    exists in the specified directory, which defaults to the local tmpdir, a
    path to that file is returned. If no such file exists, the source file is
    atomically copied, given that unique name, and a path to the copy is
    returned.

    Take care to ensure that multiple processes are not attempting to cache the
    same file to the same temporary directory!

    The caller is responsible for removing the file when appropriate.
    """

    if cache_dir == None:
        cache_dir = gettempdir()

    cached_name = "cached.%s" % uuid5(namespace, path)
    cached_path = join(cache_dir, cached_name)

    if exists(cached_path):
        return cached_path
    else:
        partial_path = "%s.partial" % cached_path

        copy2(path, partial_path)
        rename(partial_path, cached_path)

        return cached_path

def expandpath(path, relative = ""):
    """
    Expand and (possibly) extend the specified path.
    """

    return join(relative, expandvars(expanduser(path)))

def files_under(path, pattern = "*"):
    """
    Iterate over the set of paths to files in the specified directory tree.

    @param path:    Specified path.
    @param pattern: Optional filter pattern(s).
    """

    # walk the directory tree
    from os      import walk
    from os.path import (
        join,
        isfile,
        )

    if isfile(path):
        walked = [path]
    else:
        def walk_path():
            for (p, _, f) in walk(path):
                for n in f:
                    yield join(p, n)

        walked = walk_path()

    # filter names
    from fnmatch import fnmatch

    if isinstance(pattern, str):
        pattern = [pattern]

    for name in walked:
        if any(fnmatch(name, p) for p in pattern):
            yield name

def call_capturing(arguments, input = None, preexec_fn = None):
    """
    Spawn a process and return its output and status code.
    """

    popened = None

    try:
        # launch the subprocess
        import subprocess

        from subprocess import Popen

        popened = \
            Popen(
                arguments,
                stdin      = subprocess.PIPE,
                stdout     = subprocess.PIPE,
                stderr     = subprocess.PIPE,
                preexec_fn = preexec_fn,
                )

        # wait for its natural death
        (stdout, stderr) = popened.communicate(input)
    except:
        raised = Raised()

        if popened is not None and popened.poll() is None:
            try:
                popened.kill()
                popened.wait()
            except:
                Raised().print_ignored()

        raised.re_raise()
    else:
        return (stdout, stderr, popened.returncode)

def check_call_capturing(arguments, input = None, preexec_fn = None):
    """
    Spawn a process and return its output.
    """

    (stdout, stderr, code) = call_capturing(arguments, input, preexec_fn)

    if code == 0:
        return (stdout, stderr)
    else:
        from subprocess import CalledProcessError

        error = CalledProcessError(code, arguments)

        error.stdout = stdout
        error.stderr = stderr

        raise error

def unset_all(*args):
    """
    Unset every specified environment variable.
    """

    from os import environ

    for name in args:
        del environ[name]

def guess_encoding(path):
    """
    Guess a file's (compression) encoding.

    Uses the system MIME type database to detect the presence and type of
    compression encoding, if any. This method is not perfect, but should
    correctly handle common cases.
    """

    # bloody hack
    if ".xz" not in mimetypes.encodings_map:
        mimetypes.encodings_map[".xz"] = "xz"

    (mime_type, encoding) = mimetypes.guess_type(path)

    return encoding

def openz(path, mode = "r"):
    """
    Open a file, transparently [de]compressing it if a known compression extension is present.
    """

    encoding = guess_encoding(path)

    if encoding == "bzip2":
        from bz2 import BZ2File

        return BZ2File(path, mode)
    elif encoding == "gzip":
        from gzip import GzipFile

        return GzipFile(path, mode)
    elif encoding == "xz":
        raise NotImplementedError()
    elif encoding is None:
        return open(path, mode)
    else:
        raise RuntimeError("unsupported file encoding")

def xzed(bytes):
    """
    Return XZ-compressed bytes.
    """

    (stdout, _) = check_call_capturing(["xz"], bytes)

    return stdout

def unxzed(bytes):
    """
    Return XZ-decompressed bytes.
    """

    (stdout, _) = check_call_capturing(["unxz"], bytes)

    return stdout

def decompress(ipath, opath, encoding = None):
    """
    Attempt to decompress the compressed file, writing to an output path.
    """

    if encoding is None:
        encoding = guess_encoding(ipath)

    with open(opath, "w") as ofile:
        if encoding == "bzip2":
            check_call(["bunzip2", "-c", ipath], stdout = ofile)
        elif encoding == "gzip":
            check_call(["gunzip", "-c", ipath], stdout = ofile)
        elif encoding == "xz":
            check_call(["unxz", "-c", ipath], stdout = ofile)
        else:
            raise RuntimeError("uncompressed, or unsupported compression encoding")

        ofile.flush()
        fsync(ofile.fileno())

def decompress_if(ipath, opath):
    """
    Attempt to decompress the file, if compressed; return path used.
    """

    encoding = guess_encoding(ipath)

    if encoding is None:
        return ipath
    else:
        decompress(ipath, opath, encoding)

        return opath

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

def escape_for_latex(text):
    """
    Escape a text string for use in a LaTeX document.
    """

    return \
        replace_all(
            text,
            ("%", r"\%"),
            ("_", r"\_"))

@contextmanager
def mkdtemp_scoped(prefix = None):
    """
    Create, and then delete, a temporary directory.
    """

    # provide a reasonable default prefix
    if prefix is None:
        prefix = "%s." % pwd.getpwuid(os.getuid())[0]

    # create the context
    path = None

    try:
        path = tempfile.mkdtemp(prefix = prefix)

        yield path
    finally:
        if path is not None:
            shutil.rmtree(path, ignore_errors = True)

@contextmanager
def env_restored(unset = []):
    """
    Create a temporary directory, with support for cleanup.
    """

    # preserve the current environment
    from os import environ

    old = environ.copy()

    # arbitrarily modify it
    for name in unset:
        del environ[name]

    yield

    # then restore the preserved copy
    environ.clear()
    environ.update(old)

@contextmanager
def uncompressed(path):
    """
    Provide an uncompressed read-only file in a managed context.
    """

    with cargo.mkdtemp_scoped() as sandbox_path:
        sandboxed_path    = os.path.join(sandbox_path, "uncompressed")
        uncompressed_path = cargo.decompress_if(path, sandboxed_path)

        with open(uncompressed_path) as opened:
            yield opened


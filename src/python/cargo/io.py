"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

__all__ = [
    "read_named_csv",
    "write_named_csv",
    "expandvars",
    "expandpath",
    "files_under",
    "uncompressed",
    "mkdtemp_scoped",
    "call_capturing",
    "check_call_capturing",
    "openz",
    "escape_for_latex",
    ]

import os
import os.path
import bz2
import csv
import pwd
import gzip
import errno
import shutil
import hashlib
import tempfile
import threading
import mimetypes
import contextlib
import subprocess
import collections
import cargo

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
from cargo.errors import Raised

def write_named_csv(file_, tuples):
    """
    Write a list of named tuples to a CSV file.
    """

    def write(csv_file):
        writer = csv.writer(csv_file)

        if tuples:
            names = tuples[0]._fields
        else:
            return

        writer.writerow(names)
        writer.writerows(map(str, t) for t in tuples)

    if isinstance(file_, str):
        with open(file_, "w") as csv_file:
            write(csv_file)
    else:
        write(file_)

def read_named_csv(file_, Named = None, types = None, read_names = True):
    """
    Read named tuples from a CSV file.
    """

    def read(csv_file, Named, types):
        reader = csv.reader(csv_file)

        if read_names:
            names = reader.next()

            if Named is None:
                Named = collections.namedtuple("TupleFromCSV", names)
            elif set(names) != set(Named._fields):
                raise RuntimeError("names in CSV file do not match")
        else:
            names = Named._fields

        if types is None:
            types = dict((f, lambda x: x) for f in names)

        for line in reader:
            yield Named(**dict((f, types[f](v)) for (f, v) in zip(names, line)))

    if isinstance(file_, str):
        with open(file_) as csv_file:
            for row in read(csv_file, Named, types):
                yield row
    else:
        for row in read(file_, Named, types):
            yield row

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
        os.rename(partial_path, cached_path)

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
    if os.path.isfile(path):
        walked = [path]
    else:
        def walk_path():
            for (p, _, f) in os.walk(path, followlinks = True):
                for n in f:
                    yield os.path.join(p, n)

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

def openz(path, mode = "rb", closing = True):
    """Open a file, transparently [de]compressing it if a known extension is present."""

    (_, extension) = os.path.splitext(path)

    if extension == ".bz2":
        file_ = bz2.BZ2File(path, mode)
    elif extension == ".gz":
        file_ = gzip.GzipFile(path, mode)
    elif extension == ".xz":
        raise NotImplementedError()
    else:
        file_ = open(path, mode)

    if closing:
        return contextlib.closing(file_)
    else:
        return file_

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
    """Attempt to decompress the compressed file, writing to an output path."""

    (_, extension) = os.path.splitext(path)

    with open(opath, "w") as ofile:
        if extension == ".bz2":
            check_call(["bunzip2", "-c", ipath], stdout = ofile)
        elif extension == ".gz":
            check_call(["gunzip", "-c", ipath], stdout = ofile)
        elif extension == ".xz":
            check_call(["unxz", "-c", ipath], stdout = ofile)
        else:
            raise RuntimeError("uncompressed, or unsupported compression encoding")

        ofile.flush()
        os.fsync(ofile.fileno())

#def decompress_if(ipath, opath):
    #"""Attempt to decompress the file, if compressed; return path used."""

    #encoding = guess_encoding(ipath)

    #if encoding is None:
        #return ipath
    #else:
        #decompress(ipath, opath, encoding)

        #return opath

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
        cargo.replace_all(
            text,
            ("%", r"\%"),
            ("_", r"\_"))

@contextlib.contextmanager
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

@contextlib.contextmanager
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

@contextlib.contextmanager
def uncompressed(path):
    """
    Provide an uncompressed read-only file in a managed context.
    """

    with cargo.mkdtemp_scoped() as sandbox_path:
        sandboxed_path    = os.path.join(sandbox_path, "uncompressed")
        uncompressed_path = cargo.decompress_if(path, sandboxed_path)

        with open(uncompressed_path) as opened:
            yield opened


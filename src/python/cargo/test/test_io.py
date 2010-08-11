"""
cargo/test/io.py

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools import (
    eq_,
    timed,
    )

@timed(4)
def test_unxz():
    """
    Test decompression via unxz().
    """

    from cargo.io import unxzed

    data = "\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3\x01\x00\x03bar\n\x00}\x9b\x1d)J\x8ck\xc8\x00\x01\x1c\x04o,\x9c\xc1\x1f\xb6\xf3}\x01\x00\x00\x00\x00\x04YZ"

    eq_(unxzed(data), "bar\n")

@timed(4)
def test_xz():
    """
    Test compression via xz().
    """

    from cargo.io import (
        xzed,
        unxzed,
        )

    eq_("foo", unxzed(xzed("foo")))

def test_files_under():
    """
    Test directory traversal using files_under().
    """

    from cargo.io import mkdtemp_scoped

    with mkdtemp_scoped() as box_path:
        # build a fake directory tree
        from os      import makedirs
        from os.path import join

        directories = [
            join(box_path, "foo/bar/baz"),
            join(box_path, "foo/aaa/bbb"),
            join(box_path, "qux/ccc"),
            ]

        for name in directories:
            makedirs(name)

        files = [
            join(box_path, "qux/ccc/blob.a"),
            join(box_path, "qux/ccc/blub.b"),
            ]

        for name in files:
            open(name, "w").close()

        # traverse it
        from nose.tools import assert_equal
        from cargo.io   import files_under

        assert_equal(sorted(files_under(box_path)),                 sorted(files))
        assert_equal(sorted(files_under(box_path, "*.a")),          files[:1])
        assert_equal(sorted(files_under(box_path, ["*.a", "*.b"])), sorted(files))

        assert_equal(list(files_under(files[0])),       files[:1])
        assert_equal(list(files_under(directories[0])), [])


"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_save_load_json():
    """
    Test trivial JSON save and load.
    """

    from nose.tools import assert_equal
    from os.path    import join
    from cargo.io   import mkdtemp_scoped
    from cargo.json import (
        save_json,
        load_json,
        )

    with mkdtemp_scoped() as sandbox_path:
        json_path = join(sandbox_path, "test.json")
        json_data = {"foo": 42, "bar": "baz"}

        save_json(json_data, json_path)

        loaded_data = load_json(json_path)

        assert_equal(loaded_data, json_data)


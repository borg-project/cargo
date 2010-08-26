"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                     import call
    from cargo.tools.sql.describe import main

    call(main)

from plac import annotations

@annotations(
    url          = ("database URL"                 , "positional"),
    schema       = ("fully-qualified metadata name", "positional"),
    apply        = ("create the generated schema"  , "flag"       , "a"),
    alphabetical = ("sort by name"                 , "flag"       , "b"),
    quiet        = ("be less noisy"                , "flag"       , "q"),
    )
def main(url, schema = None, apply = False, alphabetical = False, quiet = False):
    """
    Print or apply a reflected or loaded database schema.
    """

    # output
    from cargo.log import (
        get_logger,
        enable_default_logging,
        )

    enable_default_logging()

    # build the particular database engine
    from cargo.sql.alchemy import make_engine

    engine = make_engine(url)

    # load the appropriate schema
    if schema is None:
        # examine the database to construct a schema
        from sqlalchemy.schema import MetaData

        metadata = MetaData(bind = engine.connect(), reflect = True)
    else:
        # load an already-defined schema
        from cargo.sugar import value_by_name

        metadata = value_by_name(schema)

    # print or apply the schema
    if apply:
        if not quiet:
            get_logger("sqlalchemy.engine", level = "DEBUG")

        metadata.create_all(engine)
    else:
        # print the DDL
        from sqlalchemy.schema import CreateTable

        if alphabetical:
            sorted_tables = sorted(metadata.sorted_tables, key = lambda t: t.name)
        else:
            sorted_tables = metadata.sorted_tables

        for table in sorted_tables:
            print CreateTable(table).compile(engine)


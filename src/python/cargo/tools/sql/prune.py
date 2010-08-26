"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                  import call
    from cargo.tools.sql.prune import main

    call(main)

from plac              import annotations
from cargo.log         import get_logger
from cargo.sql.alchemy import normalize_url

log = get_logger(__name__, level = "NOTSET")

@annotations(
    url   = ("database URL" , "positional", None, normalize_url),
    quiet = ("be less noisy", "flag"      , "q"),
    )
def main(url, quiet = False):
    """
    Drop unused tables from database.
    """

    # basic setup
    from cargo.log import enable_default_logging

    enable_default_logging()

    if not quiet:
        get_logger("sqlalchemy.engine", level = "WARNING")
        get_logger("cargo.sql.actions", level = "DETAIL")

    # connect
    from sqlalchemy import create_engine

    engine     = create_engine(url)
    connection = engine.connect()

    with connection.begin():
        # reflect the schema
        from sqlalchemy.schema import MetaData

        metadata = MetaData(bind = connection, reflect = True)

        # look for and drop empty tables
        from sqlalchemy.sql           import select
        from sqlalchemy.sql.functions import count

        for table in metadata.sorted_tables:
            ((size,),) = connection.execute(select([count()], None, table))

            if size == 0:
                log.note("table %s is empty; dropping", table.name)

                table.drop()
            else:
                log.note("table %s has %i rows", table.name, size)

    # done
    engine.dispose()


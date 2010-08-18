"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                 import call
    from cargo.tools.sql.copy import main

    call(main)

from plac      import annotations
from cargo.log import get_logger

log = get_logger(__name__, level = "NOTSET")

@annotations(
    quiet     = ("be less noisy", "flag", "q"),
    to_url    = ("destination URL"),
    from_urls = ("source URL(s)"),
    tables    = ("source URL(s)", "option", "t", lambda s: s.split(",")),
    )
def main(to_url, tables = None, quiet = False, *from_urls):
    """
    Copy data from source database(s) to some single target.
    """

    # basic setup
    from cargo.log import enable_console_log

    enable_console_log()

    if not quiet:
        get_logger("sqlalchemy.engine", level = "WARNING")
        get_logger("cargo.sql.actions", level = "DETAIL")

    # copy as requested
    from sqlalchemy        import create_engine
    from cargo.sql.alchemy import normalize_url

    to_engine     = create_engine(normalize_url(to_url))
    to_connection = to_engine.connect()

    with to_connection.begin():
        if tables is not None:
            log.debug("permitting only tables: %s", tables)

        for raw_from_url in from_urls:
            # normalize the URL
            from_url = normalize_url(raw_from_url)

            log.info("copying from %s", from_url)

            # connect to this source
            from_engine     = create_engine(from_url)
            from_connection = from_engine.connect()

            # reflect its schema
            from sqlalchemy.schema import MetaData

            metadata = MetaData(bind = from_connection, reflect = True)

            # copy its data
            for sorted_table in metadata.sorted_tables:
                if tables is None or sorted_table.name in tables:
                    log.debug("copying table %s", sorted_table.name)

                    from cargo.sql.actions import copy_table

                    copy_table(from_connection, to_connection, sorted_table)

            # done
            from_engine.dispose()

    # done
    to_engine.dispose()


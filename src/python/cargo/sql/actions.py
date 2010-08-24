"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.log import get_logger

log = get_logger(__name__)

def copy_table(from_connection, to_connection, table, where = None):
    """
    Copy a table from one connection to another.
    """

    # copy the schema, if necessary
    table.create(bind = to_connection, checkfirst = True)

    # retrieve the rows
    from sqlalchemy import select

    query = select(table.columns, whereclause = where)

    log.detail("querying via statement: %s", query)

    result = from_connection.execute(query)

    # insert the rows
    from sqlalchemy import insert

    command = table.insert(dict((c, None) for c in table.columns))

    log.detail("inserting via statement: %s", command)

    while True:
        rows = result.fetchmany(8192)

        if rows:
            log.detail("inserting %i row(s) into %s", len(rows), table)

            to_connection.execute(command, rows)
        else:
            break


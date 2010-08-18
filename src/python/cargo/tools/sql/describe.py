"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

if __name__ == "__main__":
    from plac                     import call
    from cargo.tools.sql.describe import main

    call(main)

def main(url, topological = False):
    """
    Print or apply the database schema.
    """

    # retrieve the schema
    from sqlalchemy        import create_engine
    from sqlalchemy.schema import MetaData

    engine     = create_engine(url)
    connection = engine.connect()
    metadata   = MetaData(bind = connection, reflect = True)

    # print the DDL
    from sqlalchemy.schema import CreateTable

    if topological:
        sorted_tables = metadata.sorted_tables
    else:
        sorted_tables = sorted(metadata.sorted_tables, key = lambda t: t.name)

    for table in sorted_tables:
        print CreateTable(table).compile(engine)


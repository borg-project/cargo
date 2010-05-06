"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from nose.tools        import (
    assert_equal,
    assert_raises,
    )
from cargo.sql.alchemy import with_sqlite_temporary

@with_sqlite_temporary
def test_sql_timedelta_type(engine):
    """
    Test the SQL_TimeDelta type decorator.
    """

    # database layout
    from sqlalchemy                 import (
        Column,
        Integer,
        )
    from sqlalchemy.ext.declarative import declarative_base
    from cargo.sql.alchemy          import SQL_TimeDelta

    DeclarativeBase = declarative_base()

    class DeltaRow(DeclarativeBase):
        """
        Arbitrary declared table.
        """

        __tablename__ = "deltas"

        id    = Column(Integer, primary_key = True)
        delta = Column(SQL_TimeDelta)

    DeclarativeBase.metadata.create_all(engine)

    # insert some data
    from datetime       import timedelta
    from contextlib     import closing
    from sqlalchemy.orm import sessionmaker
    from cargo.temporal import TimeDelta

    Session = sessionmaker(bind = engine)

    with closing(Session()) as session:
        session.add_all([
            DeltaRow(id = 0, delta = 6.3),
            DeltaRow(id = 1, delta = 1e6),
            DeltaRow(id = 2, delta = timedelta(12, 3, 9)),
            DeltaRow(id = 3, delta = TimeDelta(12, 3, 9)),
            DeltaRow(id = 4, delta = TimeDelta(seconds = 1e6)),
            DeltaRow(id = 5, delta = None),
            DeltaRow(id = 6, delta = 1),
            ])
        session.commit()

    # then retrieve it
    with closing(Session()) as session:
        query = session.query(DeltaRow).order_by(DeltaRow.id).all()

        assert_equal(query[0].delta, TimeDelta(seconds = 6.3))
        assert_equal(query[1].delta, TimeDelta(seconds = 1e6))
        assert_equal(query[2].delta, TimeDelta(12, 3, 9))
        assert_equal(query[3].delta, TimeDelta(12, 3, 9))
        assert_equal(query[4].delta, TimeDelta(seconds = 1e6))
        assert_equal(query[5].delta, None)
        assert_equal(query[6].delta, TimeDelta(seconds = 1))

    # try to insert some invalid data
    with closing(Session()) as session:
        session.add(DeltaRow(id = 10, delta = "baz"))

        assert_raises(TypeError, session.commit)

@with_sqlite_temporary
def test_sql_uuid_type(engine):
    """
    Test the SQL_UUID type decorator.
    """

    # database layout
    from sqlalchemy                 import (
        Column,
        Integer,
        )
    from sqlalchemy.ext.declarative import declarative_base
    from cargo.sql.alchemy          import SQL_UUID

    DeclarativeBase = declarative_base()

    class UUID_Row(DeclarativeBase):
        """
        Arbitrary declared table.
        """

        __tablename__ = "uuids"

        id   = Column(Integer, primary_key = True)
        uuid = Column(SQL_UUID)

    DeclarativeBase.metadata.create_all(engine)

    # insert some data
    from uuid           import uuid4
    from contextlib     import closing
    from sqlalchemy.orm import sessionmaker
    from cargo.temporal import TimeDelta

    Session   = sessionmaker(bind = engine)
    some_uuid = uuid4()

    with closing(Session()) as session:
        session.add_all([
            UUID_Row(id = 0, uuid = None),
            UUID_Row(id = 1, uuid = some_uuid),
            UUID_Row(id = 2, uuid = some_uuid.hex),
            ])
        session.commit()

    # then retrieve it
    with closing(Session()) as session:
        query = session.query(UUID_Row).order_by(UUID_Row.id).all()

        assert_equal(query[0].uuid, None)
        assert_equal(query[1].uuid, some_uuid)
        assert_equal(query[2].uuid, some_uuid)

    # try to insert some invalid data
    with closing(Session()) as session:
        # string of the wrong size
        session.add(UUID_Row(id = 10, uuid = "baz"))

        assert_raises(ValueError, session.commit)

        session.rollback()

        # value of the wrong type
        session.add(UUID_Row(id = 11, uuid = 1234))

        assert_raises(TypeError, session.commit)


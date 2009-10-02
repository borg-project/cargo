"""
cargo/sql/alchemy.py

Support for SQLAlchemy. As usual, we reduce boilerplate by eliminating flexibility.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import json
import datetime
import pytz
import sqlalchemy.dialects.postgresql

from uuid import UUID
from sqlalchemy import (
    Column,
    Binary,
    String,
    Integer,
    Boolean,
    DateTime,
    Interval,
    ForeignKey,
    create_engine,
    )
from sqlalchemy.orm import (
    sessionmaker,
    )
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import (
    TypeEngine,
    TypeDecorator,
    )
from sqlalchemy.dialects.postgresql.base import (
    PGUuid,
    PGArray,
    )
from cargo.flags import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.sugar import TimeDelta

SQL_Base        = declarative_base()
SQL_SessionCore = sessionmaker()
engines         = {}
module_flags    = \
    Flags(
        "SQL Configuration",
        Flag(
            "--database",
            default = "sqlite:///:memory:",
            metavar = "DATABASE",
            help    = "use DATABASE by default [%default]",
            ),
        )

def get_sql_engine(database = None):
    """
    Return the default global SQL engine.
    """

    global engine

    if database is None:
        database = module_flags.given.database

    try:
        return engines[database]
    except KeyError:
        engine = engines[database] = create_engine(database)

        create_sql_metadata(database)

        return engine

def create_sql_metadata(database = None):
    """
    Create metadata for all global SQL structures.
    """

    SQL_Base.metadata.create_all(get_sql_engine(database))

class SQL_Session(SQL_SessionCore):
    """
    Convenient but restricted SQL session construction.
    """

    def __init__(self, database = None):
        """
        Initialize.
        """

        # base
        SQL_SessionCore.__init__(self, bind = get_sql_engine(database))

class SQL_UUID(TypeDecorator):
    """
    Python-style hashable UUID column.
    """

    impl = TypeEngine

    def __init__(self):
        """
        Initialize.
        """

        TypeDecorator.__init__(self)

    def load_dialect_impl(self, dialect):
        """
        Get the dialect-specific underlying column type.
        """

        if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
            return PGUuid()
        else:
            return Binary(length = 16)

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        if value and isinstance(value, UUID):
            if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
                return value.hex
            else:
                return value.bytes
        elif value and not isinstance(value, UUID):
            raise ValueError("value %s is not a uuid.UUID" % value)
        else:
            return None

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        if value:
            if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
                return UUID(hex = value)
            else:
                return UUID(bytes = value)
        else:
            return None

    def is_mutable(self):
        """
        Are instances mutable?
        """

        return False

class UTC_DateTime(TypeDecorator):
    """
    Time zone aware (non-naive) datetime column.
    """

    impl = DateTime

    def __init__(self):
        """
        Initialize.
        """

        # members
        self.zone = pytz.utc

        # base
        TypeDecorator.__init__(self, timezone = self.zone)

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        if value and isinstance(value, datetime.datetime):
            if value.tzinfo is self.zone:
                return value
            else:
                raise ValueError("value %s is not explicitly zoned %s" % (value, self.zone.zone))
        elif value and not isinstance(value, datetime.datetime):
            raise ValueError("value %s is not a datetime instance" % value)
        else:
            return None

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

#         if value.tzinfo is not None:
#             print value.tzinfo.tzname(value)

        # FIXME
#         assert value.tzinfo is None

        if value:
            return value.replace(tzinfo = self.zone)
        else:
            return None

    def is_mutable(self):
        """
        Are instances mutable?
        """

        return False

class SQL_JSON(TypeDecorator):
    """
    Column for data structures representable as JSON strings.
    """

    impl = String

    def __init__(self):
        """
        Initialize.
        """

        # base
        TypeDecorator.__init__(self)

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        if value is None:
            return None
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        if value is None:
            return None
        else:
            return json.loads(value)

    def is_mutable(self):
        """
        Are instances mutable?
        """

        return True

class SQL_TimeDelta(TypeDecorator):
    """
    Column for data structures representable as JSON strings.
    """

    impl = Interval

    def __init__(self):
        """
        Initialize.
        """

        # base
        TypeDecorator.__init__(self)

#     def process_bind_param(self, value, dialect = None):
#         """
#         Return SQL data from a Python instance.
#         """

#         if value is None:
#             return None
#         else:
#             return json.dumps(value)

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        if value is None:
            return None
        else:
            return \
                TimeDelta(
                    days = value.days,
                    seconds = value.seconds,
                    microseconds = value.microseconds,
                    )

    def is_mutable(self):
        """
        Are instances mutable?
        """

        return False

class SQL_List(TypeDecorator):
    """
    Python-style hashable UUID column.
    """

    impl = TypeEngine

    def __init__(self, item_type):
        """
        Initialize.
        """

        # base
        TypeDecorator.__init__(self)

        # members
        self.item_type = item_type

    def load_dialect_impl(self, dialect):
        """
        Get the dialect-specific underlying column type.
        """

        if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
            return PGArray(self.item_type)
        else:
            return SQL_JSON()

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        return value

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        return value

    def is_mutable(self):
        """
        Are instances mutable?
        """

        return True


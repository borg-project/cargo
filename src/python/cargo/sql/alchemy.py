"""
cargo/sql/alchemy.py

Support for SQLAlchemy. As usual, we reduce boilerplate by eliminating flexibility.

@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import json
import datetime
import pytz
import sqlalchemy.dialects.sqlite
import sqlalchemy.dialects.postgresql

from uuid                                import UUID
from datetime                            import timedelta
from contextlib                          import contextmanager
from sqlalchemy                          import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Interval,
    ForeignKey,
    LargeBinary,
    text,
    create_engine,
    )
from sqlalchemy.orm                      import (
    sessionmaker,
    )
from sqlalchemy.ext.declarative          import declarative_base
from sqlalchemy.types                    import (
    TypeEngine,
    TypeDecorator,
    )
from sqlalchemy.dialects.postgresql.base import (
    PGUuid,
    PGArray,
    )
from nose.tools                          import make_decorator
from cargo.flags                         import (
    Flag,
    Flags,
    with_flags_parsed,
    )
from cargo.errors                        import Raised
from cargo.temporal                      import TimeDelta

def column(name, type = None):
    """
    Return a text-named column, with optional typemap, for use in SQL generation.
    """

    if type is None:
        typemap = None
    else:
        typemap = {name: type}

    return text(name, typemap = typemap)

@contextmanager
def disposing(engine):
    """
    Context manager for engine disposal.
    """

    yield engine

    engine.dispose()

def with_sqlite_temporary(callable):
    """
    Provide a safely-disposed, disk-backed SQLite engine to a nose test.
    """

    from os.path            import join
    from sqlalchemy         import create_engine
    from cargo.io           import mkdtemp_scoped
    from cargo.sql.alchemy  import disposing

    def wrapped():
        """
        The new callable.
        """

        with mkdtemp_scoped() as directory:
            url = "sqlite:///%s" % join(directory, "testing.sqlite")

            with disposing(create_engine(url, echo = False)) as engine:
                return callable(engine)

    return make_decorator(callable)(wrapped)

class SQL_Engines(object):
    """
    Manage a collection of engines.
    """

    def __init__(self):
        """
        Initialize.
        """

        self.engines = {}

    def __enter__(self):
        """
        Enter context.
        """

        return self

    def __exit__(self, *args):
        """
        Leave context.
        """

        try:
            for engine in self.engines.itervalues():
                engine.dispose()
        except:
            Raised().print_ignored()

    def get(self, database):
        """
        Return the default global SQL engine.
        """

        try:
            return self.engines[database]
        except KeyError:
            engine = self.engines[database] = create_engine(database)

            return engine

SQL_Engines.default = SQL_Engines()

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
        elif isinstance(dialect, sqlalchemy.dialects.sqlite.base.dialect):
            # using string here for convenience; suboptimal
            return String(length = 32)
        else:
            # FIXME "small binary" type?
            return LargeBinary(length = 16)

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        # convert the value into a uuid
        if value is None:
            return None
        elif isinstance(value, UUID):
            uuid_value = value
        elif isinstance(value, str):
            uuid_value = UUID(value)
        else:
            raise TypeError("value of incompatible type %s" % type(value))

        # convert the uuid into something database-compatible
        if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
            return uuid_value.hex
        elif isinstance(dialect, sqlalchemy.dialects.sqlite.base.dialect):
            return uuid_value.hex
        else:
            return uuid_value.bytes

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        if value:
            if isinstance(dialect, sqlalchemy.dialects.postgresql.base.dialect):
                return UUID(hex = value)
            elif isinstance(dialect, sqlalchemy.dialects.sqlite.base.dialect):
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

        if isinstance(value, datetime.datetime):
            if value.tzinfo is self.zone:
                return value
            else:
                raise ValueError("value %s is not explicitly zoned %s" % (value, self.zone.zone))
        elif value is None:
            return None
        else:
            raise TypeError("value of incompatible type %s" % type(value))

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

    def __init__(self, mutable = True):
        """
        Initialize.
        """

        TypeDecorator.__init__(self)

        self._mutable = mutable

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

        return self._mutable

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

    def process_bind_param(self, value, dialect = None):
        """
        Return SQL data from a Python instance.
        """

        if isinstance(value, timedelta):
            return \
                timedelta(
                    days         = value.days,
                    seconds      = value.seconds,
                    microseconds = value.microseconds,
                    )
        elif isinstance(value, float):
            return timedelta(seconds = value)
        elif isinstance(value, int):
            return timedelta(seconds = value)
        elif value is None:
            return None
        else:
            raise TypeError("value of incompatible type %s" % type(value))

    def process_result_value(self, value, dialect = None):
        """
        Return a Python instance from SQL data.
        """

        if value is None:
            return None
        else:
            return \
                TimeDelta(
                    days         = value.days,
                    seconds      = value.seconds,
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

    def __init__(self, item_type, mutable = True):
        """
        Initialize.
        """

        # base
        TypeDecorator.__init__(self)

        # members
        self.item_type = item_type
        self._mutable = mutable

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

        return self._mutable


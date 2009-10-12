# the sloppiest Makefile ever

LDFLAGS = -lpthread -lboost_python `gsl-config --libs` $(EXTRA_LDFLAGS)
# MTFLAGS = -msse2 -DDSFMT_MEXP=521 -DHAVE_SSE2 --param max-inline-insns-single=1800 --param inline-unit-growth=500 --param large-function-growth=9
# CXXFLAGS = -O3 $(MTFLAGS) -DBOOST_UBLAS_NDEBUG -DNDEBUG -I/usr/include/python2.5 -fPIC
BASEFLAGS = -g -O3 $(MTFLAGS) $(PREFIX_INCLUDE_FLAG) `python2.6-config --includes` -fPIC -Wall -Wextra -Wconversion `gsl-config --cflags`
CFLAGS = $(BASEFLAGS) -std=c99 $(EXTRA_CFLAGS)
CXXFLAGS = $(BASEFLAGS) $(EXTRA_CFLAGS)
OBJECTS = _statistics.o

all: _statistics.so _tables.so

_statistics.so: $(OBJECTS)
	gcc -O3 $(LDFLAGS) $(OBJECTS) -o _statistics.so -shared

_tables.so: _tables.o
	gcc -O3 $(LDFLAGS) _tables.o -o _tables.so -shared

clean:
	-rm -f *.o *.so


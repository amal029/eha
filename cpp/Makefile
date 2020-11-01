# Define required macros here
SHELL=/bin/bash
# SRCS=gf.cpp
OBJS = gf.o
CFLAG = -O3 -Wundefined-var-template
CC = g++ -std=c++2a
INCLUDES = -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib/ -L/usr/local/Cellar/cln/1.3.6/lib
LIBS = -lcln -lginac

gf:	${OBJS}
	export LD_LIBRARY_PATH=/usr/local/Cellar/ginac/1.8.0/lib:/usr/local/Cellar/cln/1.3.6/lib
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJS} -o $@ ${LIBS}

run:
	./gf

clean:
	-rm -f *.o core *.core gf

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} $< -c

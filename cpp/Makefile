# Define required macros here
SHELL=/bin/bash
# SRCS=gf.cpp
OBJS = robotgshs_par.o
CFLAG = -O3 -Wundefined-var-template
ifeq ($(TARGET), LINUX)
	CC = g++ -std=c++2a
else
	CC = clang++ -std=c++2a
endif
INCLUDES = -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib/ -L/usr/local/Cellar/cln/1.3.6/lib
LIBS = -lcln -lginac

robotgshs_par:	${OBJS}
	export LD_LIBRARY_PATH=/usr/local/Cellar/ginac/1.8.0/lib:/usr/local/Cellar/cln/1.3.6/lib
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJS} -o $@ ${LIBS}

run:
	./robotgshs_par

clean:
	-rm -f *.o core *.core robotgshs_par

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} $< -c

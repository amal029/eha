# Define required macros here
SHELL=/bin/bash
OBJS = robotgshs_par.o
CFLAGS = -O3 -Wundefined-var-template -mavx2 -mavx -ffast-math
ifeq ($(TARGET), LINUX)		#This is for the linux machine at work
	CC = clang++ -std=c++2a
else
	CC = g++ -std=c++2a
endif
INCLUDES = -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
INCLUDES += -I/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8

ifeq ($(TARGET), LINUX)
	INCLUDES += -I.
else
	INCLUDES += -I/Users/amal029_old/Library/Python/3.8/lib/python/site-packages/numpy/core/include
	INCLUDES += -I/Users/amal029_old/eha/cpp/include

endif

LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib/ -L/usr/local/Cellar/cln/1.3.6/lib
LIBDIR += -L/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib
LIBDIR += -L./lib
LIBS = -lcln -lginac -lpython3.8 -lsolver

robotgshs_par:	${OBJS}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJS} -o $@ ${LIBS}

run:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./robotgshs_par 1.2

clean:
	-rm -f *.o core *.core robotgshs_par

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

# This makefile is for my OSX machine only!
SHELL=/bin/bash
OBJ1 = robotgshs_par.o
OBJ2 = driver.o
OBJ3 = gene.o
OBJS = ${OBJ1} ${OBJ2} ${OBJ3}
CFLAGS = -O3 -Wundefined-var-template -mavx2 -mavx -ffast-math
CC = g++ -std=c++2a
INCLUDES = -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
INCLUDES += -I/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8

INCLUDES += -I/Users/amal029_old/Library/Python/3.8/lib/python/site-packages/numpy/core/include
INCLUDES += -I/Users/amal029_old/eha/cpp/include

LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib/ -L/usr/local/Cellar/cln/1.3.6/lib
LIBDIR += -L/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib
LIBDIR += -L./lib
LIBS = -lcln -lginac -lpython3.8 -lsolver

all:	robotgshs_par driver gene

# Example 1
robotgshs_par: ${OBJ1}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ1} -o $@ ${LIBS}

runrobot:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./robotgshs_par 1.2

# Example 2
driver: ${OBJ2}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ2} -o $@ ${LIBS}

rundriver:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./driver

# Example 3
gene: ${OBJ3}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ2} -o $@ ${LIBS}

rungene:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./gene

clean:
	-rm -f *.o *.core robotgshs_par driver all gene

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

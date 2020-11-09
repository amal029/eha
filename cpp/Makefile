# This makefile is for my OSX machine only!
SHELL=/bin/bash
OBJ1 = robotgshs_par.o
OBJ2 = driver.o
OBJ3 = gene.o
OBJ4 = twocars.o
OBJS = ${OBJ1} ${OBJ2} ${OBJ3} ${OBJ4}
CFLAGS = -O3 -mavx2 -mavx -ffast-math -std=c++2a
ifeq ($(TIME), 1)
	CFLAGS += -DTIME
endif
SYSROOT = `xcrun --show-sdk-path`
CC = clang++ -isysroot ${SYSROOT}
INCLUDES = -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
INCLUDES += -I/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8

INCLUDES += -I/Users/amal029_old/Library/Python/3.8/lib/python/site-packages/numpy/core/include
INCLUDES += -I/Users/amal029_old/eha/cpp/include

LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib -L/usr/local/Cellar/cln/1.3.6/lib
LIBDIR += -L/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib
LIBDIR += -L./lib
LIBDIR += -L/usr/local/opt/llvm/lib -L/usr/lib
LIBS = -lcln -lginac -lpython3.8 -lsolver

all:	lib robotgshs_par driver gene twocars

lib:
	cd src && make install

.PHONY:	test
test:	runrobot rundriver runtwocars rungene

# Example 1
robotgshs_par: lib ${OBJ1}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ1} -o $@ ${LIBS}

runrobot:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./robotgshs_par 1.2

# Example 2
driver: lib ${OBJ2}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ2} -o $@ ${LIBS}

rundriver:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./driver

# Example 3
gene: lib ${OBJ3}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ3} -o $@ ${LIBS}

rungene:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./gene

# Example 4
twocars: lib ${OBJ4}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ4} -o $@ ${LIBS}

runtwocars:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./twocars

clean:
	-rm -f *.o *.core robotgshs_par driver all gene twocars
	cd src && make cleanall

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

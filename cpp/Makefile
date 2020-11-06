# Define required macros here
SHELL=/bin/bash
OBJ1 = robotgshs_par.o
OBJ2 = driver.o
OBJS = ${OBJ1} ${OBJ2}
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

all:	robotgshs_par driver

# Example 1
robotgshs_par: ${OBJ1}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ1} -o $@ ${LIBS}

# Example 2
driver: ${OBJ2}
	${CC} ${CFLAGS} ${INCLUDES} ${LIBDIR} ${OBJ2} -o $@ ${LIBS}

run1:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./robotgshs_par 1.2

run2:
	DYLD_FALLBACK_LIBRARY_PATH=/Users/amal029_old/eha/cpp/lib ./driver
clean:
	-rm -f *.o *.core robotgshs_par driver all

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

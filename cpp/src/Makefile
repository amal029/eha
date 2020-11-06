OUTPUTFILE = libsolver.dylib
INSTALLDIR = ../lib
INCLUDES = -I../include
INCLUDES += -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
OBJS = solver.o
CC = g++
CFLAGS = -O3 -Wundefined-var-template -mavx2 -mavx -ffast-math -std=c++2a
LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib/ -L/usr/local/Cellar/cln/1.3.6/lib
LIBDIR += -L/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib
LIBS = -lcln -lginac -lpython3.8

.PHONY:	all
all:	${OUTPUTFILE}

.PHONY:	install
install: all
	mkdir -p ${INSTALLDIR}
	cp -p ${OUTPUTFILE} ${INSTALLDIR}

.PHONY:	clean
clean:
	rm -rf *.o *.a *.dylib *.so

.PHONY: cleanall
cleanall: clean
	rm -rf ../lib

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

${OUTPUTFILE}: ${OBJS}
	${CC} ${CFLAGS} ${LIBDIR} ${INCLUDES} -o $@ ${OBJS} -shared ${LIBS}

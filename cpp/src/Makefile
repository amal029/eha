OUTPUTFILE = libsolver.dylib
INSTALLDIR = ../lib
INCLUDES = -I../include
INCLUDES += -I/usr/local/Cellar/ginac/1.8.0/include -I/usr/local/Cellar/cln/1.3.6/include
OBJS = solver.o
SYSROOT = `xcrun --show-sdk-path`
CC = clang++ -isysroot ${SYSROOT}
CFLAGS = -O3 -mavx2 -mavx -ffast-math -std=c++2a
ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG -DZEBUG
endif
LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib -L/usr/local/Cellar/cln/1.3.6/lib
LIBDIR += -L/usr/local/Frameworks/Python.framework/Versions/Current/lib
LIBDIR += -L/usr/local/opt/llvm/lib -L/usr/lib
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

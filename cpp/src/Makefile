OUTPUTFILE = libsolver.dylib
INSTALLDIR = ../lib
INCLUDES = -I../include
INCLUDES += -I/opt/cln/include -I/opt/ginac/include
OBJS = solver.o
SYSROOT = `xcrun --show-sdk-path`
CFLAGS = -O3 -mavx2 -mavx -ffast-math
ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG -DZEBUG
endif
ifeq ($(GCC), 1)
	CC = g++-10 -std=c++20
	LIBDIR += -L/opt/ginac/lib -L/opt/cln/lib
else
	CC = clang++ -isysroot ${SYSROOT} -std=c++2a
	LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib -L/usr/local/Cellar/cln/1.3.6/lib
endif
LIBDIR += -L/usr/local/Frameworks/Python.framework/Versions/Current/lib
LIBDIR += -L/usr/local/opt/llvm/lib -L/usr/lib
# LIBDIR += -L/opt/ginac/lib -L/opt/cln/lib
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

INSTALLDIR = ../lib
INCLUDES = -I../include
OUTPUTFILE = libsolver.dll
INCLUDES += -I/opt/cln/include -I/opt/ginac/include
OBJS = solver.o
SYSROOT = `xcrun --show-sdk-path`
CFLAGS = -O2 -mavx2 -mavx -ffast-math
ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG -DZEBUG
endif
ifeq ($(GCC), 1)
	CC = g++-10 -std=c++20
	LIBDIR += -L/opt/ginac/lib -L/opt/cln/lib
else ifeq ($(MSYS), 1)
	CC = g++ -std=c++20
	CFLAGS += `pkg-config --cflags ginac`
	CFLAGS += `pkg-config --cflags cln`
	LIBDIR += `pkg-config --libs-only-L cln`
	LIBDIR += `pkg-config --libs-only-L ginac`
	LIBDIR += `pkg-config --libs-only-L gmp`
else
	CC = clang++ -isysroot ${SYSROOT} -std=c++2a
	LIBDIR = -L/usr/local/Cellar/ginac/1.8.0/lib -L/usr/local/Cellar/cln/1.3.6/lib
endif
LIBDIR += -L/usr/local/Frameworks/Python.framework/Versions/Current/lib
LIBDIR += -L/usr/local/opt/llvm/lib -L/usr/lib
# LIBDIR += `pkg-config --libs-only-L python3`
LIBDIR += -LC:/Users/amal029/AppData/Local/Programs/Python/Python39
LIBS = -lcln -lginac -lpython3.9
ifeq ($(MSYS), 1)
	LIBS += `pkg-config --libs-only-l ginac`
	LIBS += `pkg-config --libs-only-other ginac`
	LIBS += `pkg-config --libs-only-l gmp`
endif

.PHONY:	all
all:	${OUTPUTFILE}

.PHONY:	install
install: all
	mkdir -p ${INSTALLDIR}
	cp -p ${OUTPUTFILE} ${INSTALLDIR}

.PHONY:	clean
clean:
	rm -rf *.o *.a *.dylib *.so *.dll

.PHONY: cleanall
cleanall: clean
	rm -rf ../lib

.cpp.o:
	${CC} ${CFLAGS} ${INCLUDES} $< -c

ifeq ($(MSYS), 1)
${OUTPUTFILE}: ${OBJS}
	ar rcs -o $@ solver.o
else
${OUTPUTFILE}: ${OBJS}
	${CC} ${CFLAGS} ${LIBDIR} ${INCLUDES} -o $@ ${OBJS} -shared ${LIBS}
endif

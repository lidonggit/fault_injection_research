#prefix =	/usr/local
#exec_prefix =	${prefix}
#bindir =	$(exec_prefix)/bin
#libdir =	$(exec_prefix)/lib

CC =		gcc
AR =		ar
# For use with BSD ar (MacOSX ar) edit the ARFLAGS to be -s
#ARFLAGS =	@ARFLAGS@
RM =		rm
RANLIB =	ranlib
DEFS =		-DHAVE_CONFIG_H
LIBS =
INCLUDES =	-I.
CFLAGS =	-g -O2 $(INCLUDES)
LDFLAGS =


all: libfault_injector.so libfault_injector.a

fault_injector_dyn.o: fault_injector.c fault_injector.h
	$(CC) $(CFLAGS) -fPIC -c fault_injector.c -o $@

fault_injector.o: fault_injector.c fault_injector.h
	$(CC) $(CFLAGS) -c fault_injector.c

libfault_injector.a: fault_injector.o
	$(AR) $(ARFLAGS) $@ fault_injector.o

libfault_injector.so: fault_injector_dyn.o
	$(CC) $(CFLAGS) -shared fault_injector_dyn.o -o $@


#common.o: $(srcdir)/common.c $(srcdir)/common.h
#	$(CC) $(CFLAGS) -c $(srcdir)/common.c

clean:
	rm -f *.o libfault_injector.a
	rm -f libfault_injector.so


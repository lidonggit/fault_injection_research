include ./make.header

# Default target is CUDA.
OPENARC_ARCH ?= 0

# C compiler
ifeq ($(OPENARC_ARCH),0)
# CUDA target
CC = gcc
CPP = g++
BACKENDCPP = nvcc
COMMONCFLAGS = -D NVIDIA_GPU $(GCUDA_FLAGS)
COMMONCLIBS = $(GCUDA_LIBS) -lOpenCL -lcuda
COMMONCINCLUDE = $(GCUDA_INCLUDE)
OPENARCLIB_SUFFIX = cuda
CC_FLAGS_PREFIX = -Xcompiler
else
# OpenCL target
CC = gcc
CPP = g++
BACKENDCPP = g++
COMMONCFLAGS = $(GOPENCL_FLAGS)
COMMONCLIBS = $(GOPENCL_LIBS)
COMMONCINCLUDE = $(GOPENCL_INCLUDE)
OPENARCLIB_SUFFIX = opencl
CC_FLAGS_PREFIX =
endif

# Archive option
AR = ar
ARFLAGS = rs

TARGET = .

ifeq ($(OMP),1)
CLIBS = $(COMMONCLIBS) -lm -lgomp
OMP_FLAGS = -fopenmp 
OUTPUT_BASE = libopenaccrtomp_$(OPENARCLIB_SUFFIX)
else
CLIBS = $(COMMONCLIBS) -lm
OUTPUT_BASE = libopenaccrt_$(OPENARCLIB_SUFFIX)
endif

MODE ?= normal
ifeq ($(MODE),profile)
CC_FLAGS = -D _OPENARC_PROFILE_=1 $(CC_FLAGS_PREFIX) $(OMP_FLAGS)
BENCHMARK = $(TARGET)/$(OUTPUT_BASE)pf.a
else
CC_FLAGS =$(CC_FLAGS_PREFIX) $(OMP_FLAGS)
BENCHMARK = $(TARGET)/$(OUTPUT_BASE).a
endif

RESLIB = $(TARGET)/libresilience.a

PRINT_LOG ?= 0
OMP ?= 0
DEFSET ?= -DPRINT_LOG=$(PRINT_LOG) -DOMP=$(OMP)


ifeq ($(OPENARC_ARCH),0)
OBJS = openacc.o openaccrt.o opencldriver.o cudadriver.o
else
OBJS = openacc.o openaccrt.o opencldriver.o
endif

all: $(OBJS)
	$(AR) $(ARFLAGS) $(BENCHMARK) $(OBJS)

res: resilience.o
	ar rs $(RESLIB) resilience.o

openacc.o: openacc.cpp openacc.h openaccrt.h
	$(BACKENDCPP) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o openacc.o openacc.cpp -c

openaccrt.o: openaccrt.cpp openacc.h openaccrt.h resilience.cpp
	$(BACKENDCPP) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o openaccrt.o openaccrt.cpp -c


opencldriver.o: opencldriver.cpp openacc.h openaccrt.h resilience.cpp
	$(BACKENDCPP) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o opencldriver.o opencldriver.cpp -c

cudadriver.o: cudadriver.cpp openacc.h openaccrt.h resilience.cpp
	$(BACKENDCPP) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o cudadriver.o cudadriver.cpp -c

resilience.o: resilience.cpp
	$(CPP) $(DEFSET) -o resilience.o resilience.cpp -c
	
binUtil: binBuilder.cpp
	$(BACKENDCPP) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) $(CLIBS) -o $(OPENARC_BIN)/binBuilder_$(OPENARCLIB_SUFFIX) binBuilder.cpp

clean:
	rm -f *.o *~

purge:
	rm -f *.o *~ libopenaccrt_$(OPENARCLIB_SUFFIX).a libopenaccrtomp_$(OPENARCLIB_SUFFIX).a libopenaccrt_$(OPENARCLIB_SUFFIX)pf.a libopenaccrtomp_$(OPENARCLIB_SUFFIX)pf.a binBuilder_$(OPENARCLIB_SUFFIX) libresilience.a Timer

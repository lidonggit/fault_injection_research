 #
 # $Id: Makefile,v 1.3 1999/08/03 04:41:42 alc Exp $
 #
include ../make.header
FC	= g77
FFLAGS  = -O2
CXX = g++
CC = gcc
NVCC = nvcc
#CFLAGS2  =  $(GNVCC_FLAGS_NOOMP)
#CFLAGS  = -O3 -W -fopenmp -ffloat-store #same result as -g
#CFLAGS  = -O3 -W -ffloat-store #same result as -g
#CFLAGS  = -O2 -W #different from -g
#CFLAGS  = -O1 -W #same result as -g
#CFLAGS  = -g -W
#CXXFLAGS = -fno-strict-aliasing
#TARGET = .
TARGET = ./bin
BENCHMARK_SERIAL = ./lulesh_serial
BENCHMARK_SERIAL2 = $(TARGET)/lulesh_serial
BENCHMARK_NV = $(TARGET)/lulesh_ARC_NV
BENCHMARK_AMD = $(TARGET)/lulesh_ARC_AMD

ifeq ($(OMP),1)
CLIBS1_BASE = -lm -lgomp
CLIBS2_BASE = -lm -lgomp
lulesh_serial: CFLAGS1_BASE = -fopenmp
lulesh_ARC_NV: CFLAGS2_BASE = -Xcompiler -fopenmp
lulesh_ARC_AMD: CFLAGS2_BASE = -fopenmp
ACCRT_BASE = openaccrtomp
else
CLIBS1_BASE = -lm
CLIBS2_BASE = -lm 
ACCRT_BASE = openaccrt
endif

MODE ?= normal
ifeq ($(MODE), profile)
PREF = pf
endif


#Select underlying compiler-specific runtime library and compile flags
lulesh_ARC_AMD: ACCRT_BASE2 = $(ACCRT_BASE)_opencl$(PREF)
lulesh_ARC_AMD: CFLAGS2 = $(CFLAGS2_BASE) -I$(OPENARCLIB) $(AMD_INCLUDE) $(OPENCL_FLAGS) -I$(HOME)/local/include $(AMD_INCLUDE)
#lulesh_ARC_AMD: CLIBS2 = $(CLIBS2_BASE) -L$(OPENARCLIB) -l$(ACCRT_BASE2) $(OPENCL_LIB) -L$(HOME)/local/lib -ljemalloc
lulesh_ARC_AMD: CLIBS2 = $(CLIBS2_BASE) -L$(OPENARCLIB) -l$(ACCRT_BASE2) $(OPENCL_LIB)

lulesh_ARC_NV: ACCRT_BASE2 = $(ACCRT_BASE)_cuda$(PREF)   
lulesh_ARC_NV: CFLAGS2 = $(CFLAGS2_BASE) -I$(OPENARCLIB) $(CUDA_INCLUDE) $(GNVCC_FLAGS) -I$(HOME)/local/include -DDEBUG
#lulesh_ARC_NV: CLIBS2 = $(CLIBS2_BASE) -L$(OPENARCLIB) -l$(ACCRT_BASE2) $(OPENCL_LIB) $(CUDA_LIB) -L$(HOME)/local/lib -ljemalloc
lulesh_ARC_NV: CFLAGS2 = $(CFLAGS2_BASE) -I$(OPENARCLIB) $(CUDA_INCLUDE) $(GNVCC_FLAGS)

lulesh_serial: ACCRT_BASE1 = resilience
lulesh_serial: CFLAGS1  = $(CFLAGS1_BASE) -O3 -fno-strict-aliasing -DUNIX -fPIC -ffloat-store -I$(HOME)/local/include
#lulesh_serial: CLIBS1 = $(CLIBS1_BASE) -L$(HOME)/local/lib -ljemalloc -L$(OPENARCLIB) -l$(ACCRT_BASE1)
lulesh_serial: CLIBS1 = $(CLIBS1_BASE) -L$(OPENARCLIB) -l$(ACCRT_BASE1)
lulesh_serial2: ACCRT_BASE1 = resilience
lulesh_serial2: CFLAGS1  = $(CFLAGS1_BASE) -O3 -fno-strict-aliasing -DUNIX -fPIC -ffloat-store -I$(HOME)/local/include -I$(OPENARCLIB)
#lulesh_serial2: CLIBS1 = $(CLIBS1_BASE) -L$(HOME)/local/lib -ljemalloc -L$(OPENARCLIB) -l$(ACCRT_BASE1)
lulesh_serial2: CLIBS1 = $(CLIBS1_BASE) -L$(OPENARCLIB) -l$(ACCRT_BASE1)

serial2: lulesh_serial2
serial: lulesh_serial
ARC_NV: lulesh_ARC_NV
ARC_AMD: lulesh_ARC_AMD
all_NV: lulesh_serial lulesh_ARC_NV
all_AMD: lulesh_serial lulesh_ARC_AMD

lulesh_serial: lulesh.c
	if [ ! -d $(TARGET) ]; then mkdir $(TARGET); fi
	$(CXX) $(CFLAGS1) -o $(BENCHMARK_SERIAL) lulesh.c $(CLIBS1)

lulesh_serial2: $(CETUS_OUTPUT)/lulesh.c
	if [ ! -d $(TARGET) ]; then mkdir $(TARGET); fi
	cd $(CETUS_OUTPUT); $(CXX) $(CFLAGS1) -o ../$(BENCHMARK_SERIAL2) lulesh.c $(CLIBS1); cd ../

lulesh_ARC_NV: $(CETUS_OUTPUT)/lulesh.cpp
	if [ ! -d $(TARGET) ]; then mkdir $(TARGET); fi
	cd $(CETUS_OUTPUT); $(NVCC) $(CFLAGS2) -o ../$(BENCHMARK_NV) lulesh.cpp $(CLIBS2); cp openarc_kernel.cu $(TARGET); cd ../

lulesh_ARC_AMD: $(CETUS_OUTPUT)/lulesh.cpp
	if [ ! -d $(TARGET) ]; then mkdir $(TARGET); fi
	cd $(CETUS_OUTPUT); $(CXX) $(CFLAGS2) -o ../$(BENCHMARK_AMD) lulesh.cpp $(CLIBS2); cp openarc_kernel.cl $(TARGET); cd ../

clean:
	rm -f *.o;
	if [ -d ./$(CETUS_OUTPUT) ]; then cd $(CETUS_OUTPUT); rm -f *.o *~; cd ..; fi
	if [ -d $(TARGET) ]; then rm -f $(BENCHMARK_SERIAL) $(BENCHMARK_SERIAL2) $(BENCHMARK_NV) $(BENCHMARK_AMD); cd $(TARGET); rm -f openarc_kernel*.ptx openarc_kernel.cu openarc_kernel.cl binBuilder_cuda binBuilder_opencl; fi

purge: clean
	if [ -d ./$(CETUS_OUTPUT) ]; then cd $(CETUS_OUTPUT); rm -f *.cpp *.h *.c *.cl *.cu; cd ..; fi

#Set USE_NVIDIA to 0 if OpenCL program runs on non-NVIDIA GPUs.
include ../make.header

# C compiler
CC = g++ 
NVCC = nvcc
CC_FLAGS_SERIAL = -O3 -Wall -ffloat-store 
CLIBS1 = -lm 
#TARGET is where the output binary is stored
TARGET = ./bin

#Set the output binary names for different compilers
BENCHMARK_SERIAL = $(TARGET)/matmul_serial
BENCHMARK_OPENCL = $(TARGET)/matmul_ARC_OPENCL
BENCHMARK_CUDA = $(TARGET)/matmul_ARC_CUDA


#Choose the correct runtime library depending upon whether OMP is enabled or not.
ifeq ($(OMP),1)
CLIBS2_BASE = -lm -lgomp
matmul_ARC_CUDA: CFLAGS2_BASE = -Xcompiler -fopenmp
matmul_ARC_OPENCL: CFLAGS2_BASE = -fopenmp
ACCRT_BASE = openaccrtomp
else
CLIBS2_BASE = -lm 
CFLAGS2_BASE = 
ACCRT_BASE = openaccrt
endif

#Check for the profiler mode
MODE ?= normal
ifeq ($(MODE), profile)
PREF = pf
else
PREF = 
endif

ifeq ($(ASPEN), aspenrt)
ASPENFILE = "aspenrt.cpp"
else 
ASPENFILE = 
endif

#Select underlying compiler-specific runtime library and compile flags
matmul_ARC_OPENCL: ACCRT_BASE2 = $(ACCRT_BASE)_opencl$(PREF)
matmul_ARC_OPENCL: CFLAGS2 = $(CFLAGS2_BASE) -I$(OPENARCLIB) $(OPENCL_INCLUDE) $(OPENCL_FLAGS)
matmul_ARC_OPENCL: CLIBS2 = $(CLIBS2_BASE) -L$(OPENARCLIB)/ -l$(ACCRT_BASE2) $(OPENCL_LIB)

matmul_ARC_CUDA: ACCRT_BASE2 = $(ACCRT_BASE)_cuda$(PREF)	
matmul_ARC_CUDA: CFLAGS2 = $(CFLAGS2_BASE) -I$(OPENARCLIB) $(CUDA_INCLUDE) $(GNVCC_FLAGS)
matmul_ARC_CUDA: CLIBS2 = $(CLIBS2_BASE) -L$(OPENARCLIB)/ -l$(ACCRT_BASE2) $(OPENCL_LIB) $(CUDA_LIB)


ARC_CUDA: matmul_ARC_CUDA
ARC_OPENCL: matmul_ARC_OPENCL
all: all_CUDA
all_CUDA: matmul_ARC_CUDA matmul_serial
all_OPENCL: matmul_ARC_OPENCL matmul_serial
serial: matmul_serial

#########
# Macro #
#########
_N_ ?= 4096
OMP ?= 0
DEFSET_SERIAL ?= -D_N_=$(_N_) -DOMP=$(OMP)
DEFSET_ACC ?= -D_N_=$(_N_) -DOMP=$(OMP)

matmul_serial: matmul.c 
	$(CC) $(COMMONFLAGS) $(CC_FLAGS_SERIAL) $(DEFSET_SERIAL) -o $(BENCHMARK_SERIAL) matmul.c $(CLIBS1)

matmul_ARC_CUDA: $(CETUS_OUTPUT)/matmul.cpp
	cd $(CETUS_OUTPUT); $(NVCC) $(CFLAGS2) $(DEFSET_ACC) -o ../$(BENCHMARK_CUDA) matmul.cpp $(ASPENFILE) $(CLIBS2); cp openarc_kernel.cu ../$(TARGET); cp $(OPENARCLIB)/binBuilder_cuda ../$(TARGET); cp $(OPENARCLIB)/Timer ../$(TARGET); cp $(OPENARCLIB)/resilience.cu ../$(TARGET); cp $(OPENARCLIB)/resilience.h ../$(TARGET); cd ../


matmul_ARC_OPENCL: $(CETUS_OUTPUT)/matmul.cpp
	cd $(CETUS_OUTPUT); $(OPENCL_CC) $(CFLAGS2) $(DEFSET_ACC) -o ../$(BENCHMARK_OPENCL) matmul.cpp $(ASPENFILE) $(CLIBS2); cp openarc_kernel.cl ../$(TARGET); cp $(OPENARCLIB)/binBuilder_opencl ../$(TARGET); cp $(OPENARCLIB)/Timer ../$(TARGET); cp $(OPENARCLIB)/resilience.cl ../$(TARGET); cp $(OPENARCLIB)/resilience.h ../$(TARGET); cd ../

clean:
	rm -f *.o;
	if [ -d ./$(CETUS_OUTPUT) ]; then cd $(CETUS_OUTPUT); rm -f *.o *~ memcheck aspenrt memcheck.c aspenrt.c; cd ..; fi
	if [ -d $(TARGET) ]; then rm -f $(BENCHMARK_SERIAL) $(BENCHMARK_CUDA) $(BENCHMARK_OPENCL) Timer; cd $(TARGET); rm -f openarc_kernel*.ptx openarc_kernel.cu openarc_kernel.cl binBuilder_cuda binBuilder_opencl Timer resilience.*; fi

purge: clean
	if [ -d ./$(CETUS_OUTPUT) ]; then cd $(CETUS_OUTPUT); rm -f *.cpp *.h *.cl *.cu *.aspen *.c *.log aspenrt memcheck smallmodeltest; cd ..; fi

-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
OpenARC 0.2 (June 23, 2014)

Open Accelerator Research Compiler (OpenARC) is a framework built on top of 
the Cetus compiler infrastructure (http://cetus.ecn.purdue.edu), which is 
written in Java for C.
OpenARC provides extensible environment, where various performance 
optimizations, traceability mechanisms, fault tolerance techniques, etc., 
can be built for better debuggability/performance/resilience on the complex 
accelerator computing. 
OpenARC supports the full feature set of OpenACC V1.0 and performs 
source-to-source transformations, targeting heterogeneous devices, such as 
NVIDIA GPUs and Intel MICs.


-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* JAVA SE 6
* GCC
* ANTLRv2 
	- Default antlr.jar file is included in this distribution (./lib)

 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
* Build OpenARC runtime
  - To compile the output program that OpenARC translated from the input OpenACC
  program, OpenARC runtime should be compiled too. (refer to 
  readme_openarcrt.txt in openarcrt directory.)


-------------------------------------------------------------------------------
ENVIRONMENT SETUP
(CF: for CPU fault injection tests, this section can be skipped.)
-------------------------------------------------------------------------------
* Environment variable, OPENARC_ARCH, is used to set a target architecture, 
for which OpenARC translates the input OpenACC program. 
(Default target is NVIDIA CUDA if the variable does not exist.) 
  - Set OPENARC_ARCH = 0 for CUDA (default)
                       1 for OpenCL (e.g., AMD GPUs)
                       2 for OpenCL for Xeon Phi
  - For example in BASH, 
    export OPENARC_ARCH=0

* To port OpenACC to non-CUDA devices, OpenACC environment variables,
ACC_DEVICE_TYPE, should be set to the target device type.
  - For example in BASH, if target device is an AMD GPU,
    export ACC_DEVICE_TYPE=RADEON 

* OpenMP environment variable, OMP_NUM_THREADS, shoud be set to the maximum
number of OpenMP threads that the input program uses, if OpenMP is used in
the input OpenACC program.

* Environment variable, OPENARC_JITOPTION, may be optinally used to pass
options to the backend runtime compiler (NVCC compiler options for JIT CUDA 
kernel compilation or clBuildProgram options for JIT OpenCL kernel compilation).
  - For example, if output OpenCL kernel file (openarc_kernel.cl) contains
  header files, path to the header files may need to be specified to the backend
  OpenCL compiler.
    export OPENARC_JITOPTION="-I ."


-------------------------------------------------------------------------------
RUNNING OpenARC
-------------------------------------------------------------------------------
Users can run OpenARC in the following way:

  $ java -classpath=<user_class_path> openacc.exec.ACC2GPUDriver <options> <C files>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for OpenARC users.


-------------------------------------------------------------------------------
TESTING
-------------------------------------------------------------------------------
"./test" directory contains examples showing how to use OpenARC.
* To use scripts in the test directory:
	1) enter test directory and copy "make.header.sample" file to "make.header".
	$ cd ./test
	$ cp make.header.sample make.header
	2) Update make.header file as necessary.
	   (for CPU fault injection tests, only OPENARCLIB needs to be updated.)


-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features

* Updates

* Bug fixes and improvements

* Updates in flags


-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
This OpenARC release has the following contents.

  readme_openarc.txt     - This file
  lib                    - Archived classes (jar)
  build.sh               - Command line build script
  build.xml              - Build configuration for Apache Ant
  batchCleanup.bash      - Global cleanup script
  src                    - OpenARC source code
  doc                    - OpenARC documents
  openaccrt              - OpenARC runtime (HeteroIR) source code
  test                   - Examples showing how to use OpenARC


-------------------------------------------------------------------------------
LIMITATIONS
-------------------------------------------------------------------------------
- The underlying C parser in the current implementation supports C99 
features only partially. If parsing errors occur for the valid input 
C program, the program may contain unsuppported C99 features.
    - One of C99 features not fully supported in the current implementation 
	is mixed declaration and code; to fix this, put all variable declaration 
	statements in a function before all code sections in the function body.
        - Example: change "int a; a = 1; int b;" to "int a; int b; a = 1;".

- C preprocessor in the current implementation does not expand macros 
in pragma annotations. To enable this, use "#pragma acc #define macro value" 
directive.

- Class member is not allowed in an OpenACC subarray, and the start index 
of a subarray should be 0 (partial array passing is allowed only if its start 
index is 0.)

- Current implementation ignores vector clause. (e.g., for CUDA target, 
gang clause is used to set the number of thread blocks, and worker clause is 
used to specify the number of  threads in a thread block.)
 
- Current implementation allows data regions to have compute regions 
interprocedurally, but a compute region can have a function call only if 
the called fuction does not contain any OpenACC loop directive.  


June 23, 2014
The OpenARC Team

URL: http://ft.ornl.gov/research/openarc
EMAIL: lees2@ornl.gov

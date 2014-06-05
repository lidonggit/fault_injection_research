-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
OpenARC Runtime 0.2 (June 23, 2014)

OpenARC Runtime implements APIs used by the output program translated
by OpenARC.


-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* GCC
* NVCC to run on the CUDA target
* GCC or other OpenCL compiler to run on the OpenCL target

 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
* Build
  - Copy "make.header.sample" to "make.header", and modify environment 
  variables in the "make.header" file according to user's environment.
  - Run "batchmake.bash"
    $ ./batchmake.bash
  - CF: for CPU fault injection tests, just copy "make.header.sample" to
    "make.header", and run "make res"
	$ make res


-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features

* Updates

* Bug fixes and improvements


-------------------------------------------------------------------------------
LIMITATIONS
-------------------------------------------------------------------------------


June 23, 2014
The OpenARC Team

URL: http://ft.ornl.gov/research/openarc
EMAIL: lees2@ornl.gov

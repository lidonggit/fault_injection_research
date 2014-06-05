#include "openaccrt.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
////////////////////////////////////////////////////////
//Current implementation works with CUDA5.0 or later. //
////////////////////////////////////////////////////////

//std::map<std::string, CUfunction> CudaDriver::kernelMap;
std::vector<std::string> CudaDriver::kernelNameList;
std::map<CUdeviceptr,int> CudaDriver::pinnedHostMemCounter;

///////////////////////////
// Device Initialization //
///////////////////////////
CudaDriver::CudaDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::CudaDriver()\n");
	}
#endif
    dev = devType;
    device_num = devNum;
	num_devices = numDevices;
//Moved to init()
/*
    cudaDeviceProp deviceProp;
    cudaError_t cuResult = cudaGetDeviceProperties(&deviceProp, device_num);

    int thread_id = get_thread_id();
    fprintf(stderr, "CUDA : Host Thread %d initializes device %d: %s\n", thread_id, device_num, deviceProp.name);
*/

    for (std::vector<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        //kernelMap[*it]= 0;
        CudaDriver::kernelNameList.push_back(*it);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::CudaDriver()\n");
	}
#endif
}

HI_error_t CudaDriver::init() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::init()\n");
	}
#endif

    CUresult err;
    int major, minor;
    cudaDeviceProp deviceProp;
    cudaError_t cuResult = cudaGetDeviceProperties(&deviceProp, device_num);

    int thread_id = get_thread_id();
    fprintf(stderr, "CUDA : Host Thread %d initializes device %d: %s\n", thread_id, device_num, deviceProp.name);
    cuDeviceGet(&cuDevice, device_num);

#if CUDA_VERSION >= 5000
    cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
#else
    cuDeviceComputeCapability(&major, &minor, cuDevice);
#endif

    std::stringstream ss;
    ss << major;
    //fprintf(stderr, "Version no. major %s\n", str.c_str());*/

    ss << minor;
    std::string version = ss.str();

    std::string ptxName = std::string("openarc_kernel_") + version + std::string(".ptx");

    //compile a PTX if it does not already exist
    if( access( ptxName.c_str(), F_OK ) == -1 ) {
        std::string command = std::string("nvcc $OPENARC_JITOPTION -arch=sm_") + version + std::string(" openarc_kernel.cu -ptx -o ") + ptxName;
        //fprintf(stderr, "Version no. %s\n", command.c_str());
        system(command.c_str());
    }


	//Default flag (0) uses CU_CTX_SCHED_AUTO, but to make cuCtxSynchronize()
	//blocking, CU_CTX_SCHED_BLOCKING_SYNC should be used instead.
    //err = cuCtxCreate(&cuContext, 0, cuDevice);
    err = cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::init()] failed to create CUDA context with error %d (NVIDIA CUDA GPU)\n", err);
    }

    std::string ptx_source;
    FILE *fp = fopen(ptxName.c_str(), "rb");
    if(fp == NULL) {
        fprintf(stderr, "[ERROR in CudaDriver::init()] failed to open PTX file %s in CUDA (NVIDIA CUDA GPU)\n", ptxName.c_str());
        //printf("PTX not openend\n");
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *buf = new char[file_size+1];
    fseek(fp, 0, SEEK_SET);
    fread(buf, sizeof(char), file_size, fp);
    fclose(fp);
    buf[file_size] = '\0';
    ptx_source = buf;
    delete[] buf;

    //PTX JIT
    const unsigned int jitNumOptions = 2;
    CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
    void **jitOptVals = new void*[jitNumOptions];

    // set up size of compilation log buffer
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;

    // set up pointer to set the Maximum # of registers for a particular kernel
    /*jitOptions[2] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = 32;
    jitOptVals[2] = (void *)(size_t)jitRegCount;

    jitOptions[2] = CU_JIT_TARGET;
    int nullVal = 0;
    jitOptVals[2] = (void *)(uintptr_t)CU_TARGET_COMPUTE_30;
    */

    err = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::init()] Module Load FAIL\n");
    }

    CUstream s0, s1;
    cuStreamCreate(&s0, 0);
    queueMap[0] = s0;
    cuStreamCreate(&s1, 0);
    queueMap[1] = s1;

    CUevent e0, e1;
    std::map<int, CUevent> eventMap;
    cuEventCreate(&e0, CU_EVENT_DEFAULT);
    eventMap[0]= e0;
    cuEventCreate(&e1, CU_EVENT_DEFAULT);
    eventMap[1]= e1;
    threadQueueEventMap[get_thread_id()] = eventMap;

    createKernelArgMap();

    init_done = 1;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::init()\n");
	}
#endif
    return HI_success;
}

HI_error_t CudaDriver::createKernelArgMap() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::createKernelArgMap()\n");
	}
#endif
    CUresult err;
    cuCtxSetCurrent(cuContext);
    std::map<std::string, argmap_t*> kernelArgs;
    std::map<std::string, CUfunction> kernelMap;
    for(std::vector<std::string>::iterator it=CudaDriver::kernelNameList.begin(); it!=CudaDriver::kernelNameList.end(); ++it) {
        // Create argument mapping for the kernel
        const char *kernelName = (*it).c_str();
        CUfunction cuFunc;
        argmap_t argMap;
        kernelArgs.insert(std::pair<std::string, argmap_t*>(std::string(kernelName), new argmap_t));
        err = cuModuleGetFunction(&cuFunc, cuModule, kernelName);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "[ERROR in CudaDriver::createKernelArgMap()] Function Load FAIL on %s\n", kernelName);
        }
        kernelMap[*it] = cuFunc;
    }

    HostConf_t * tconf = getHostConf();
    tconf->kernelArgsMap[this] = kernelArgs;
    tconf->kernelsMapCUDA[this]=kernelMap;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::createKernelArgMap()\n");
	}
#endif
    return HI_success;
}

int CudaDriver::HI_get_num_devices(acc_device_t devType) {
    int numDevices;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_get_num_devices()\n");
	}
#endif
    cudaGetDeviceCount(&numDevices);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_get_num_devices()\n");
	}
#endif
    return numDevices;
}


HI_error_t CudaDriver::destroy() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::destroy()\n");
	}
#endif
    CUresult err = cuCtxDestroy(cuContext);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::destroy()] failed to destroy CUDA context with error %d (NVIDIA CUDA GPU)\n", err);
        return HI_error;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::destroy()\n");
	}
#endif
    return HI_success;
}

// Pin host memory
void CudaDriver::pin_host_memory(const void* hostPtr, size_t size)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::pin_host_memory()\n");
	}
#endif
#ifdef _OPENMP
    #pragma omp critical
#endif
    {
        CUdeviceptr host = (CUdeviceptr)hostPtr;
        //If the hostPtr is already pinned
        if(CudaDriver::pinnedHostMemCounter.find(host) != CudaDriver::pinnedHostMemCounter.end() && CudaDriver::pinnedHostMemCounter[host] > 0)	{
            CudaDriver::pinnedHostMemCounter[host]++;
        } else {

            CUresult cuResult = cuMemHostRegister((void*)host, size, CU_MEMHOSTREGISTER_PORTABLE);
            if(cuResult == CUDA_SUCCESS) {
                CudaDriver::pinnedHostMemCounter[host] = 1;
                //fprintf(stderr, "[ERROR in HI_malloc1D()] Succesfully pin the host memory\n");
            } else	{
#ifdef _OPENMP
                fprintf(stderr, "[ERROR in pin_host_memory()] Cannot pin host memory with error %d tid: %d\n", cuResult, omp_get_thread_num());
#else
                fprintf(stderr, "[ERROR in pin_host_memory()] Cannot pin host memory with error %d tid: %d\n", cuResult, 0);
#endif
            }

        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::pin_host_memory()\n");
	}
#endif
}

void CudaDriver::unpin_host_memory(const void* hostPtr)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::unpin_host_memory()\n");
	}
#endif
#ifdef _OPENMP
    #pragma omp critical
#endif
    {
        CUdeviceptr host = (CUdeviceptr)hostPtr;
        //If the hostPtr is already pinned
        if(CudaDriver::pinnedHostMemCounter.find(host) != CudaDriver::pinnedHostMemCounter.end()) {
            if(CudaDriver::pinnedHostMemCounter[host] > 1) {
                CudaDriver::pinnedHostMemCounter[host]--;
            } else
            {
                CUresult cuResult = cuMemHostUnregister((void*)host);
                if(cuResult == CUDA_SUCCESS){
                	//CudaDriver::pinnedHostMemCounter[host] = 0;
                	CudaDriver::pinnedHostMemCounter.erase(host);
                } else {
                	fprintf(stderr, "[ERROR in unpin_host_memory()] Cannot unpin host memory with error %d\n", cuResult);
                }
            }
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::unpin_host_memory()\n");
	}
#endif
}

HI_error_t  CudaDriver::HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_malloc1D(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();
    if( tconf == NULL ) {
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
#else
        int thread_id = 0;
#endif
        fprintf(stderr, "[ERROR in HI_malloc1D()] No host configuration exists for the current host thread (thread ID: %d); please set an environment variable, OMP_NUM_THREADS, to the maximum number of OpenMP threads used for your application; exit!\n", thread_id);
        exit(1);
    }
    if( tconf->device->init_done == 0 ) {
        tconf->HI_init();
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_error;


    if(HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success ) {
        result = HI_success;
    } else {
        CUresult cuResult = cuMemAlloc((CUdeviceptr*)devPtr, (size_t) count);
        if( cuResult == CUDA_SUCCESS ) {
            //Pin host memory
            pin_host_memory(hostPtr, (size_t) count);

            HI_set_device_address(hostPtr, *devPtr, asyncID);
#ifdef _OPENARC_PROFILE_
            tconf->DMallocCnt++;
#endif
            result = HI_success;
        } else {
			//[DEBUG] CUresult and cudaError_t do not match.
            //fprintf(stderr, "[ERROR in CudaDriver::HI_malloc1D()] CUDA memory alloc failed with error %d %s\n", cuResult, cudaGetErrorString((cudaError_t)cuResult));
            fprintf(stderr, "[ERROR in CudaDriver::HI_malloc1D()] CUDA memory alloc failed with error %d\n", cuResult);
        }
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_malloc1D(%d)\n", asyncID);
	}
#endif
    return result;
}

//the ElementSizeBytes in cuMemAllocPitch is currently set to 16.
HI_error_t CudaDriver::HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_malloc2D(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    if( tconf->device->init_done == 0 ) {
        tconf->HI_init();
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result;

    if(HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success ) {
        result = HI_success;
    } else {
        CUresult cuResult = cuMemAllocPitch((CUdeviceptr*)devPtr, pitch, widthInBytes, height, 16);
        if( cuResult == CUDA_SUCCESS ) {
            //Pin host memory
            pin_host_memory(hostPtr, (size_t) widthInBytes*height);

            HI_set_device_address(hostPtr, *devPtr, asyncID);
#ifdef _OPENARC_PROFILE_
            tconf->DMallocCnt++;
#endif
            result = HI_success;
        } else {
			//[DEBUG] CUresult and cudaError_t do not match.
            //fprintf(stderr, "[ERROR in CudaDriver::HI_malloc1D()] CUDA memory alloc failed with error %d %s\n", cuResult, cudaGetErrorString((cudaError_t)cuResult));
            fprintf(stderr, "[ERROR in CudaDriver::HI_malloc1D()] CUDA memory alloc failed with error %d\n", cuResult);
        }
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_malloc2D(%d)\n", asyncID);
	}
#endif
    return result;
}


HI_error_t CudaDriver::HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_malloc3D(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    //TODO
    HI_error_t result;
    result = HI_error;
#ifdef _OPENARC_PROFILE_
    tconf->DMallocCnt++;
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_malloc3D(%d)\n", asyncID);
	}
#endif
    return result;
}



HI_error_t CudaDriver::HI_free( const void *hostPtr, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_free(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    HI_error_t result = HI_success;
    void *devPtr;
    //Check if the mapping exists. Free only if a mapping is found
    if( HI_get_device_address(hostPtr, &devPtr, asyncID) != HI_error) {
        CUresult cuResult = cuMemFree((CUdeviceptr)(devPtr));
        if( cuResult == CUDA_SUCCESS ) {
            HI_remove_device_address(hostPtr, asyncID);
            // Unpin host memory
            unpin_host_memory(hostPtr);

#ifdef _OPENARC_PROFILE_
            tconf->DFreeCnt++;
#endif
        } else {
			//[DEBUG] CUresult and cudaError_t do not match.
            //fprintf(stderr, "[ERROR in CudaDriver::HI_free()] CUDA memory free failed with error %d %s\n", cuResult, cudaGetErrorString((cudaError_t)cuResult));
            fprintf(stderr, "[ERROR in CudaDriver::HI_free()] CUDA memory free failed with error %d\n", cuResult);
            result = HI_error;
        }
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_free(%d)\n", asyncID);
	}
#endif
    return result;
}




//malloc used for allocating temporary data.
//If the method is called for a pointer to existing memory, the existing memory
//will be freed before allocating new memory.
void CudaDriver::HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_tempMalloc1D()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    if( devType == acc_device_gpu || devType == acc_device_nvidia || 
		devType == acc_device_radeon || devType == acc_device_current) {
		if( tempMallocSet.count(*tempPtr) > 0 ) {
			tempMallocSet.erase(*tempPtr);	
            //cudaFree(*tempPtr);
    		CUresult cuResult = cuMemFree((CUdeviceptr)*tempPtr);
    		if(cuResult != CUDA_SUCCESS) {
        		fprintf(stderr, "[ERROR in CudaDriver::HI_tempMalloc1D()] failed to free on CUDA with error %d (NVIDIA CUDA GPU)\n", cuResult);
    		}
#ifdef _OPENARC_PROFILE_
            tconf->DFreeCnt++;
#endif
		}
        //cudaMalloc(tempPtr, count);
    	CUresult cuResult = cuMemAlloc((CUdeviceptr*)tempPtr, (size_t) count);
    	if(cuResult != CUDA_SUCCESS) {
        	fprintf(stderr, "[ERROR in CudaDriver::HI_tempMalloc1D()] failed to malloc on CUDA with error %d (NVIDIA CUDA GPU)\n", cuResult);
    	}
		tempMallocSet.insert(*tempPtr);	
#ifdef _OPENARC_PROFILE_
        tconf->DMallocCnt++;
#endif
    } else {
		if( tempMallocSet.count(*tempPtr) > 0 ) {
			tempMallocSet.erase(*tempPtr);	
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->HFreeCnt++;
#endif
        }
        *tempPtr = malloc(count);
		tempMallocSet.insert(*tempPtr);	
#ifdef _OPENARC_PROFILE_
        tconf->HMallocCnt++;
#endif
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_tempMalloc1D()\n");
	}
#endif
}

//Used for de-allocating temporary data.
void CudaDriver::HI_tempFree( void** tempPtr, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_tempFree()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    if( devType == acc_device_gpu || devType == acc_device_nvidia 
    || devType == acc_device_radeon || devType == acc_device_current ) {
        if( *tempPtr != 0 ) {
			tempMallocSet.erase(*tempPtr);	
            //cudaFree(*tempPtr);
    		CUresult cuResult = cuMemFree((CUdeviceptr)*tempPtr);
    		if(cuResult != CUDA_SUCCESS) {
        		fprintf(stderr, "[ERROR in CudaDriver::HI_tempFree()] failed to free on CUDA with error %d (NVIDIA CUDA GPU)\n", cuResult);
    		}
#ifdef _OPENARC_PROFILE_
            tconf->DFreeCnt++;
#endif
        }
    } else {
        if( *tempPtr != 0 ) {
			tempMallocSet.erase(*tempPtr);	
            free(*tempPtr);
    		// Unpin host memory if already pinned.
    		unpin_host_memory(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->HFreeCnt++;
#endif
        }
    }
    *tempPtr = 0;
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_tempFree()\n");
	}
#endif
}


//////////////////////
// Kernel Execution //
//////////////////////


//In the driver API, copying into a constant memory (symbol) does not require a different API call
HI_error_t  CudaDriver::HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();

    CUresult cuResult;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    switch( kind ) {
    case HI_MemcpyHostToHost: {
        cuResult = cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, count);
        break;
    }
    case HI_MemcpyHostToDevice: {
        cuResult = cuMemcpyHtoD((CUdeviceptr) dst, src, count);
        break;
    }
    case HI_MemcpyDeviceToHost: {
        cuResult = cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
        break;
    }
    case HI_MemcpyDeviceToDevice: {
        cuResult = cuMemcpyDtoD((CUdeviceptr) dst, (CUdeviceptr)src, count);
        break;
    }
    }
#ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
        tconf->H2DMemTrCnt++;
        tconf->H2DMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToHost ) {
        tconf->D2HMemTrCnt++;
        tconf->D2HMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
        tconf->D2DMemTrCnt++;
        tconf->D2DMemTrSize += count;
    } else {
        tconf->H2HMemTrCnt++;
        tconf->H2HMemTrSize += count;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( cuResult == CUDA_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy()\n");
	}
#endif
        return HI_success;
    } else {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HI_memcpy()] Memcpy failed with error %d in tid %d\n", cuResult, omp_get_thread_num());
		//[DEBUG] CUresult and cudaError_t do not match.
        //fprintf(stderr, "Error messages: %s\n", cudaGetErrorString((cudaError_t)cuResult));
#else
        fprintf(stderr, "[ERROR in HI_memcpy()] Memcpy failed with error %d in tid %d\n", cuResult, 0);
		//[DEBUG] CUresult and cudaError_t do not match.
        //fprintf(stderr, "Error messages: %s\n", cudaGetErrorString((cudaError_t)cuResult));
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy()\n");
	}
#endif
        return HI_error;
    }
}

HI_error_t CudaDriver::HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy_const()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    CUresult cuResult;
    HI_error_t result = HI_success;
    CUdeviceptr dptr;
    size_t size;
    cuResult = cuModuleGetGlobal( &dptr, &size, cuModule, constName.c_str());

//#ifdef _OPENARC_PROFILE_
//	double ltime = HI_get_localtime();
//#endif

    if( cuResult != CUDA_SUCCESS ) {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HI_memcpy_const()] Acquiring constant memory handle failed with error %d in tid %d\n", cuResult, omp_get_thread_num());
#else
        fprintf(stderr, "[ERROR in HI_memcpy_const()] Acquiring constant memory handle failed with error %d in tid %d\n", cuResult, 0);
#endif
        result = HI_error;
    }

    result = HI_memcpy((void*)dptr, hostPtr, count, kind, 0);

//#ifdef _OPENARC_PROFILE_
//    tconf->totalMemTrTime += HI_get_localtime() - ltime;
//#endif

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy_const()\n");
	}
#endif
    return result;
}


HI_error_t CudaDriver::HI_memcpy_async(void *dst, const void *src, size_t count,
        HI_MemcpyKind_t kind, int trType, int async) {
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy_async(%d)\n", async);
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    CUresult cuResult;
    CUstream stream = getQueue(async);
    CUevent event = getEvent(async);

    switch( kind ) {
    case HI_MemcpyHostToHost: {
        cuResult = cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, count);
        break;
    }
    case HI_MemcpyHostToDevice: {
        cuResult = cuMemcpyHtoDAsync((CUdeviceptr) dst, src, count, stream);
        break;
    }
    case HI_MemcpyDeviceToHost: {
        cuResult = cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream);
        break;
    }
    case HI_MemcpyDeviceToDevice: {
        cuResult = cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr)src, count, stream);
        break;
    }
    }

    cuEventRecord(event, stream);
#ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
        tconf->H2DMemTrCnt++;
        tconf->H2DMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToHost ) {
        tconf->D2HMemTrCnt++;
        tconf->D2HMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
        tconf->D2DMemTrCnt++;
        tconf->D2DMemTrSize += count;
    } else {
        tconf->H2HMemTrCnt++;
        tconf->H2HMemTrSize += count;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( cuResult == CUDA_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy_async(%d)\n", async);
	}
#endif
        return HI_success;
    } else {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HI_memcpy_async()] Memcpy failed with error %d in tid %d with asyncId %d\n", cuResult, omp_get_thread_num(), async);
#else
        fprintf(stderr, "[ERROR in HI_memcpy_async()] Memcpy failed with error %d in tid %d with asyncId %d\n", cuResult, 0, async);
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy_async(%d)\n", async);
	}
#endif
        return HI_error;
    }
}

//Used for kernel verification.
HI_error_t CudaDriver::HI_memcpy_asyncS(void *dst, const void *src, size_t count,
        HI_MemcpyKind_t kind, int trType, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    CUresult cuResult;
    CUstream stream = getQueue(async);
    CUevent event = getEvent(async);

    switch( kind ) {
    case HI_MemcpyHostToHost: {
        cuResult = cuMemcpy((CUdeviceptr) dst, (CUdeviceptr) src, count);
        break;
    }
    case HI_MemcpyHostToDevice: {
        cuResult = cuMemcpyHtoDAsync((CUdeviceptr) dst, src, count, stream);
        break;
    }
    case HI_MemcpyDeviceToHost: {
		void *tDst = 0;
		HI_tempMalloc1D(&tDst, count, acc_device_host);
        //Pin host memory
        pin_host_memory(tDst, (size_t) count);
		HI_set_temphost_address(dst, tDst, async);
        cuResult = cuMemcpyDtoHAsync(tDst, (CUdeviceptr)src, count, stream);
        break;
    }
    case HI_MemcpyDeviceToDevice: {
        cuResult = cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr)src, count, stream);
        break;
    }
    }

    cuEventRecord(event, stream);
#ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
        tconf->H2DMemTrCnt++;
        tconf->H2DMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToHost ) {
        tconf->D2HMemTrCnt++;
        tconf->D2HMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
        tconf->D2DMemTrCnt++;
        tconf->D2DMemTrSize += count;
    } else {
        tconf->H2HMemTrCnt++;
        tconf->H2HMemTrSize += count;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( cuResult == CUDA_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
        return HI_success;
    } else {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] Memcpy failed with error %d in tid %d with asyncId %d\n", cuResult, omp_get_thread_num(), async);
#else
        fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] Memcpy failed with error %d in tid %d with asyncId %d\n", cuResult, 0, async);
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
        return HI_error;
    }
}


HI_error_t CudaDriver::HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
        size_t widthInBytes, size_t height, HI_MemcpyKind_t kind) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy2D()\n");
	}
	
#endif
    HostConf_t * tconf = getHostConf();

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    CUresult cuResult=CUDA_ERROR_DEINITIALIZED;
    CUDA_MEMCPY2D pcopy;
    switch( kind ) {
    case HI_MemcpyHostToHost: {
        pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
        pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
        pcopy.srcHost = src;
        pcopy.dstHost = dst;
        break;
    }
    case HI_MemcpyHostToDevice: {
        pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
        pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
        pcopy.srcHost = src;
        pcopy.dstDevice = (CUdeviceptr) dst;
        break;
    }
    case HI_MemcpyDeviceToHost: {
        pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
        pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
        pcopy.srcDevice = (CUdeviceptr) src;
        pcopy.dstHost = dst;
        break;
    }
    case HI_MemcpyDeviceToDevice: {
        pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
        pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
        pcopy.srcDevice = (CUdeviceptr) src;
        pcopy.dstDevice = (CUdeviceptr) dst;
        break;
    }
    }

    pcopy.srcXInBytes = 0;
    pcopy.srcY = 0;
    pcopy.dstXInBytes = 0;
    pcopy.dstY = 0;
    pcopy.srcPitch = spitch;
    pcopy.dstPitch = dpitch;
    pcopy.WidthInBytes = widthInBytes;
    pcopy.Height = height;

    cuResult = cuMemcpy2D(&pcopy);
    //fprintf(stderr, "[in HI_memcpy2D()] Memcpy done\n");
#ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
        tconf->H2DMemTrCnt++;
        tconf->H2DMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToHost ) {
        tconf->D2HMemTrCnt++;
        tconf->D2HMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
        tconf->D2DMemTrCnt++;
        tconf->D2DMemTrSize += widthInBytes*height;
    } else {
        tconf->H2HMemTrCnt++;
        tconf->H2HMemTrSize += widthInBytes*height;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( cuResult == CUDA_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy2D()\n");
	}
#endif
        return HI_success;
    } else {
				fprintf(stderr, "[ERROR in HI_memcpy2D()] Memcpy failed with error %d \n", cuResult);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy2D()\n");
	}
#endif
        return HI_error;
    }
}

HI_error_t CudaDriver::HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_memcpy2D_async(%d)\n", async);
	}
#endif
    HostConf_t * tconf = getHostConf();
    /*
    #ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
    #endif
    CUresult cuResult;
    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    cudaStream_t stream;
    asyncmap_t * asyncmap = tconf->asyncmaptable;

    if( asyncmap->count(async) > 0 ) {
    	stream = asyncmap->at(async);
    } else {
    	cudaStreamCreate(&stream);
    	(*asyncmap)[async] = stream;
    }

    void * dstT;
    const void * srcT;
    void * pinnedHostPtr;
    size_t count;
    if( kind == HI_MemcpyHostToHost ) {
    	CUDA_MEMCPY2D pcopy;
    	pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
    	pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
    	pcopy.srcHost = src;
    	pcopy.dstHost = dst;
    	pcopy.srcXInBytes = 0;
    	pcopy.srcY = 0;
    	pcopy.dstXInBytes = 0;
    	pcopy.dstY = 0;
    	pcopy.srcPitch = spitch;
    	pcopy.dstPitch = dpitch;
    	pcopy.WidthInBytes = widthInBytes;
    	pcopy.Height = height;
    	cuResult = cuMemcpy2D(&pcopy);
    } else {
    	if( kind == HI_MemcpyHostToDevice ) {
    		//cuResult = cudaMallocHost(&pinnedHostPtr, count);
    		count = spitch*height;
    		pinnedHostPtr = tconf->get_asyncphostaddress(devNum, async, src, count);
    		if( tconf->has_pendingphost2hostcopy(devNum, async, src) == 0 ) {
    			//cuResult = cudaMemcpy(pinnedHostPtr, src, count, cudaMemcpyHostToHost);
    			cuResult = cuMemcpyHtoD((CUdeviceptr)pinnedHostPtr, src, count);
    		}
    		srcT = pinnedHostPtr;
    		dstT = dst;
    	} else if( kind == HI_MemcpyDeviceToHost ) {
    		//cuResult = cudaMallocHost(&pinnedHostPtr, count);
    		count = dpitch*height;
    		pinnedHostPtr = tconf->get_asyncphostaddress(devNum, async, dst, count);
    		dstT = pinnedHostPtr;
    		srcT = src;
    	} else {
    		dstT = dst;
    		srcT = src;
    	}

    	CUDA_MEMCPY2D pcopy;
    	switch( kind ) {
    		case HI_MemcpyHostToHost: {pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
    									  pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
    									  pcopy.srcHost = src;
    									  pcopy.dstHost = dst;}
    		case HI_MemcpyHostToDevice: {pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
    										pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
    										pcopy.srcHost = src;
    										pcopy.dstDevice = (CUdeviceptr) dst;}
    		case HI_MemcpyDeviceToHost: {pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
    										pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
    										pcopy.srcDevice = (CUdeviceptr) src;
    										pcopy.dstHost = dst;}
    		case HI_MemcpyDeviceToDevice: {pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
    										  pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
    										  pcopy.srcDevice = (CUdeviceptr) src;
    										  pcopy.dstDevice = (CUdeviceptr) dst;}
    	}

    	pcopy.srcXInBytes = 0;
    	pcopy.srcY = 0;
    	pcopy.dstXInBytes = 0;
    	pcopy.dstY = 0;
    	pcopy.srcPitch = spitch;
    	pcopy.dstPitch = dpitch;
    	pcopy.WidthInBytes = widthInBytes;
    	pcopy.Height = height;

    	cuResult = cuMemcpy2DAsync(&pcopy, stream);

    	//cuResult = cudaMemcpy2DAsync(dstT, dpitch, srcT, spitch, widthInBytes, height, toCudaMemcpyKind(kind), stream);
    	if( kind == HI_MemcpyHostToDevice ) {
    		//cuResult = cudaFreeHost(pinnedHostPtr);
    		tconf->set_asynchostphostmap(devNum, async, src, pinnedHostPtr);
    	} else if( kind == HI_MemcpyDeviceToHost ) {
    		//cuResult = cudaMemcpy(dst, pinnedHostPtr, count, cudaMemcpyHostToHost);
    		tconf->set_asynchostsizemap(devNum, async, dst, count);
    		//cuResult = cudaFreeHost(pinnedHostPtr);
    		tconf->set_asynchostphostmap(devNum, async, dst, pinnedHostPtr);
    	}
    }
    #ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
    	tconf->H2DMemTrCnt++;
    	tconf->H2DMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToHost ) {
    	tconf->D2HMemTrCnt++;
    	tconf->D2HMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
    	tconf->D2DMemTrCnt++;
    	tconf->D2DMemTrSize += widthInBytes*height;
    } else {
    	tconf->H2HMemTrCnt++;
    	tconf->H2HMemTrSize += widthInBytes*height;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
    #endif
    if( cuResult == CUDA_SUCCESS ) {
    	return HI_success;
    } else {
    	return HI_error;
    }

    */
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_memcpy2D_async(%d)\n", async);
	}
#endif
}


HI_error_t CudaDriver::HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_register_kernel_arg()\n");
	}
#endif
    argument_t arg;
    HostConf_t *tconf = getHostConf();
    std::map<std::string, argmap_t*> kernelargs = tconf->kernelArgsMap.at(this);
    argmap_t* argumentMap = kernelargs.at(kernel_name);

    arg.arg_val = arg_value;
    arg.arg_size = arg_size;

    argumentMap->insert(std::pair<int, argument_t>(arg_index, arg));

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_register_kernel_arg()\n");
	}
#endif
    return HI_success;
}



HI_error_t CudaDriver::HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_kernel_call(%d)\n", async);
	}
#endif
    HostConf_t *tconf = getHostConf();
    std::map<std::string, argmap_t*> kernelargs = tconf->kernelArgsMap.at(this);
    argmap_t* argumentMap = kernelargs.at(kernel_name);
    size_t argumentCount = argumentMap->size();
    CUresult err;
    void** kernelArgs = (void**)malloc(sizeof(void*) * argumentCount);

    for(int i = 0; i < argumentCount; i++)
    {
        argument_t arg = argumentMap->at(i);
        kernelArgs[i] = arg.arg_val;
    }
    //fprintf(stderr, "[HI_kernel_call()] GRIDSIZE %d %d %d\n", gridSize[2], gridSize[1], gridSize[0]);
    CUfunction kernel = tconf->kernelsMapCUDA.at(this).at(kernel_name);
    if(async != DEFAULT_QUEUE) {
        CUstream stream = getQueue(async);
        CUevent event = getEvent(async);
        err = cuLaunchKernel(kernel, gridSize[0], gridSize[1], gridSize[2], blockSize[0], blockSize[1], blockSize[2], 0, stream, kernelArgs, NULL);

        cuEventRecord(event, stream);

    } else {
        err = cuLaunchKernel(kernel, gridSize[0], gridSize[1], gridSize[2], blockSize[0], blockSize[1], blockSize[2], 0, 0, kernelArgs, NULL);
    }
    if (err != CUDA_SUCCESS) {
        free(kernelArgs);
        free(kernelArgs);
		//[DEBUG] CUresult and cudaError_t do not match.
        //fprintf(stderr, "[ERROR in CudaDriver::HI_kernel_call()] Kernel [%s] Launch FAIL with error %d %s\n",kernel_name.c_str(), err, cudaGetErrorString((cudaError_t)err));
        fprintf(stderr, "[ERROR in CudaDriver::HI_kernel_call()] Kernel [%s] Launch FAIL with error %d\n",kernel_name.c_str(), err);
        return HI_error;
    }

    free(kernelArgs);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_kernel_call(%d)\n", async);
	}
#endif
    return HI_success;
}

HI_error_t CudaDriver::HI_synchronize()
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_synchronize()\n");
	}
#endif
    CUresult err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
		//[DEBUG] CUresult and cudaError_t do not match.
        //fprintf(stderr, "[ERROR in CudaDriver::HI_synchronize()] Context Synchronization FAIL with error %d %s\n", err, cudaGetErrorString((cudaError_t)err));
        fprintf(stderr, "[ERROR in CudaDriver::HI_synchronize()] Context Synchronization FAIL with error %d\n", err);
        return HI_error;
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_synchronize()\n");
	}
#endif
    return HI_success;
}



HI_error_t CudaDriver::HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_bind_tex()\n");
	}
#endif

    CUresult err;
    CUtexref cuTexref;
    HI_error_t result = HI_success;
    err = cuModuleGetTexRef(&cuTexref, cuModule, texName.c_str());
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to find CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
    }
    err = cuTexRefSetAddress(0, cuTexref, (CUdeviceptr)devPtr, size);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to set address for CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
    }
    err = cuTexRefSetAddressMode(cuTexref, 0, CU_TR_ADDRESS_MODE_WRAP);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to set address mode for CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
    }
    err = cuTexRefSetFilterMode(cuTexref, CU_TR_FILTER_MODE_LINEAR);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to set filter mode for CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
    }
    err = cuTexRefSetFlags(cuTexref, CU_TRSF_NORMALIZED_COORDINATES);
    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to set flags for CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
    }

    if(type == HI_int) {
        err = cuTexRefSetFormat(cuTexref, CU_AD_FORMAT_SIGNED_INT32, 1);
    } else if (type == HI_float) {
        err = cuTexRefSetFormat(cuTexref, CU_AD_FORMAT_FLOAT, 1);
    } else {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] Unsupported format for CUDA texture '%s' (NVIDIA CUDA GPU)\n", texName.c_str());
        result = HI_error;
    }

    if(err != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_bind_tex()] failed to set format for CUDA texture '%s' with error %d (NVIDIA CUDA GPU)\n", texName.c_str(), err);
        result = HI_error;
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_bind_tex()\n");
	}
#endif
    return result;
}

void CudaDriver::HI_set_async(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_set_async(%d)\n", asyncId);
	}
#endif
#ifdef _OPENMP
    #pragma omp critical
#endif
    {
        asyncId += 2;
        std::map<int, CUstream >::iterator it= queueMap.find(asyncId);

        if(it == queueMap.end()) {
            CUstream str;
            cuStreamCreate(&str, 0);
            queueMap[asyncId] = str;
        }

        int thread_id = get_thread_id();
        std::map<int, std::map<int, CUevent> >::iterator threadIt;
        threadIt = threadQueueEventMap.find(thread_id);

        //threadQueueEventMap is empty for this thread
        if(threadIt == threadQueueEventMap.end()) {
            std::map<int, CUevent> newMap;
            CUevent ev;
            cuEventCreate(&ev, CU_EVENT_DEFAULT);
            newMap[asyncId] = ev;
            threadQueueEventMap[thread_id] = newMap;
        } else {
            //threadQueueEventMap does not have an entry for this stream
            //std::map<int, CUevent> evMap = threadIt->second;
            if(threadIt->second.find(asyncId) == threadIt->second.end()) {
                CUevent ev;
                cuEventCreate(&ev, CU_EVENT_DEFAULT);
                threadIt->second[asyncId] = ev;
                //threadIt->second = evMap;
            }
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_set_async(%d)\n", asyncId-2);
	}
#endif
}

void CudaDriver::HI_async_wait(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_wait(%d)\n", asyncId);
	}
#endif
    CUevent event = getEvent(asyncId);
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    CUresult cuResult = cuEventSynchronize(event);

    if(cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_async_wait()] failed wait on CUDA queue %d with error %d (NVIDIA CUDA GPU)\n", asyncId, cuResult);
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif

    HI_postponed_free(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_wait(%d)\n", asyncId);
	}
#endif
}

void CudaDriver::HI_async_waitS1(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_waitS1(%d)\n", asyncId);
	}
#endif
    CUevent event = getEvent(asyncId);
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    CUresult cuResult = cuEventSynchronize(event);

    if(cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_async_wait()] failed wait on CUDA queue %d with error %d (NVIDIA CUDA GPU)\n", asyncId, cuResult);
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_waitS1(%d)\n", asyncId);
	}
#endif
}

void CudaDriver::HI_async_waitS2(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_waitS2(%d)\n", asyncId);
	}
#endif
	HI_free_temphosts(asyncId);
    HI_postponed_free(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_waitS2(%d)\n", asyncId);
	}
#endif
}

void CudaDriver::HI_async_wait_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_wait_all()\n");
	}
#endif
    eventmap_cuda_t eventMap = threadQueueEventMap.at(get_thread_id());
    HostConf_t * tconf = getHostConf();
    CUresult cuResult;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    std::set<int> queuesChecked;

    for(eventmap_cuda_t::iterator it = eventMap.begin(); it != eventMap.end(); ++it) {
        cuResult = cuEventSynchronize(it->second);
        if(cuResult != CUDA_SUCCESS) {
            fprintf(stderr, "[ERROR in CudaDriver::HI_async_wait_all()] failed wait on CUDA queue %d with error %d (NVIDIA CUDA GPU)\n", it->first, cuResult);
        }
        queuesChecked.insert(it->first);
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif

    //release the waiting frees
    std::set<int>::iterator it;
    for (it=queuesChecked.begin(); it!=queuesChecked.end(); ++it) {
        HI_postponed_free(*it);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_wait_all()\n");
	}
#endif
}

int CudaDriver::HI_async_test(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
    CUevent event = getEvent(asyncId);
    HostConf_t * tconf = getHostConf();

    CUresult cuResult = cuEventQuery(event);

    if(cuResult != CUDA_SUCCESS) {
        //fprintf(stderr, "in CudaDriver::HI_async_test()] stream %d code %d\n", asyncId, cuResult);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
        return 0;
    }

    HI_postponed_free(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
    return 1;
}

int CudaDriver::HI_async_test_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_async_test_all()\n");
	}
#endif
    eventmap_cuda_t eventMap = threadQueueEventMap.at(get_thread_id());
    HostConf_t * tconf = getHostConf();
    CUresult cuResult;

    std::set<int> queuesChecked;

    for(eventmap_cuda_t::iterator it = eventMap.begin(); it != eventMap.end(); ++it) {
        cuResult = cuEventQuery(it->second);
        if(cuResult != CUDA_SUCCESS) {
            return 0;
        }
        queuesChecked.insert(it->first);
    }

    //release the waiting frees
    std::set<int>::iterator it;
    for (it=queuesChecked.begin(); it!=queuesChecked.end(); ++it) {
        HI_postponed_free(*it);
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_async_test_all()\n");
	}
#endif
    return 1;
}


void CudaDriver::HI_malloc(void **devPtr, size_t size) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_malloc()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    CUresult cuResult = cuMemAlloc((CUdeviceptr*)devPtr, (size_t) size);
    if(cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_malloc()] failed to malloc on CUDA with error %d (NVIDIA CUDA GPU)\n", cuResult);
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_malloc()\n");
	}
#endif
}


void CudaDriver::HI_free(void *devPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter CudaDriver::HI_free()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    CUresult cuResult = cuMemFree((CUdeviceptr)devPtr);

    if(cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "[ERROR in CudaDriver::HI_free()] failed to free on CUDA with error %d (NVIDIA CUDA GPU)\n", cuResult);
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit CudaDriver::HI_free()\n");
	}
#endif
}

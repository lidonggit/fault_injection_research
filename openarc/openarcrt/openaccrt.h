#ifndef __OPENARC_HEADER__

#define __OPENARC_HEADER__

#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <stdio.h>

#ifdef NVIDIA_GPU
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#include "openacc.h"

#include <string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#if OMP == 1
#include <omp.h>
#endif

#define DEFAULT_QUEUE -2
#define DEFAULT_ASYNC_QUEUE -1

typedef enum {
    HI_success = 0,
    HI_error = 1
} HI_error_t;

typedef enum {
    HI_MemcpyHostToHost = 0,
    HI_MemcpyHostToDevice = 1,
    HI_MemcpyDeviceToHost = 2,
    HI_MemcpyDeviceToDevice = 3
} HI_MemcpyKind_t;

typedef enum {
    HI_notstale = 0,
    HI_stale = 1,
    HI_maystale = 2
} HI_memstatus_t;

typedef enum {
    HI_int = 0,
    HI_float = 1,
} HI_datatype_t;


typedef std::map<const void *, void *> addressmap_t;
typedef std::map<const void *, int> countermap_t;
typedef std::map<const void *, size_t> sizemap_t;
typedef std::map<int, addressmap_t *> asyncphostmap_t;
typedef std::map<int, sizemap_t *> asynchostsizemap_t;
typedef std::map<const void *, HI_memstatus_t> memstatusmap_t;

typedef std::multimap<int, std::map<const void *, void *> > addresstable_t;
typedef std::multimap<int, const void *> asyncfreetable_t;

#ifdef NVIDIA_GPU
typedef std::map<int, cudaStream_t> asyncmap_t;
typedef cudaStream_t HI_async_handle_t;
typedef std::map<int, CUevent> eventmap_cuda_t;
#endif

typedef std::map<int, cl_event> eventmap_opencl_t;
typedef std::set<const void *> pointerset_t;
typedef std::map<int, pointerset_t *> asyncfreemap_t;

typedef class HostConf HostConf_t;
typedef struct
{
    size_t arg_size;
    void* arg_val;
} argument_t;
typedef std::map<int, argument_t> argmap_t;
typedef class Accelerator
{
public:
    // Device info
    acc_device_t dev;
    int device_num;
	int num_devices;
    int init_done;
    //Host-device address mapping table, augmented with stream id
    addresstable_t masterAddressTable;

	//temporarily allocated memory set.
	pointerset_t tempMallocSet;
    
    //Host-TempHost address mapping table, augmented with stream id
    addresstable_t tempHostAddressTable;

    //This table can have duplicate entries, owing to the HI_free_async
    //calls in a loop. To handle this, HI_free ensures that on a duplicate
    //pair, no free operation is performed
    asyncfreetable_t postponedFreeTable;

    // Kernel Initialization
    virtual HI_error_t init() = 0;
    virtual HI_error_t destroy()=0;

    // Kernel Execution
    virtual HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value) = 0;
    virtual HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE) = 0;
    virtual HI_error_t HI_synchronize( )=0;

    // Memory Allocation
    virtual HI_error_t  HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID)= 0;
    virtual HI_error_t HI_malloc2D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID)=0;
    virtual HI_error_t HI_malloc3D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID)=0;
    virtual HI_error_t HI_free( const void *host_ptr, int asyncID)=0;

    // Memory Transfer
    virtual HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType)=0;

    virtual HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind)=0;
    virtual HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async)=0;

    virtual void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType)=0;
    virtual void HI_tempFree( void** tempPtr, acc_device_t devType)=0;

    virtual HI_error_t createKernelArgMap() {
        return HI_success;
    }
    virtual HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size) {
        return HI_success;
    }
    virtual HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count) {
        return HI_success;
    }
    virtual void HI_set_async(int asyncId)=0;
    virtual void HI_async_wait(int asyncId) {}
    virtual void HI_async_waitS1(int asyncId) {}
    virtual void HI_async_waitS2(int asyncId) {}
    virtual void HI_async_wait_all() {}
    virtual int HI_async_test(int asyncId)=0;
    virtual int HI_async_test_all()=0;

    virtual void HI_malloc(void **devPtr, size_t size) = 0;
    virtual void HI_free(void *devPtr) = 0;

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            *devPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = masterAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
                *devPtr = it2->second;
                return  HI_success;
            }
            //fprintf(stderr, "[ERROR in get_device_address()] No mapping found for the host pointer\n");
            return HI_error;
        }
    }

    HI_error_t HI_set_device_address(const void *hostPtr, void * devPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        //fprintf(stderr, "[in set_device_address()] Setting address\n");
        if(it == masterAddressTable.end() ) {
            //fprintf(stderr, "[in set_device_address()] No mapping found for the asyncID\n");
            std::map<const void *,void*> emptyMap;
            masterAddressTable.insert(std::pair<int, std::map<const void *,void*> > (asyncID, emptyMap));
            it = masterAddressTable.find(asyncID);
        }

        //(it->second).insert(std::pair<const void *,void*>(hostPtr, devPtr));
        (it->second)[hostPtr] = devPtr;
        return  HI_success;
    }

    HI_error_t HI_remove_device_address(const void *hostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);

        if(it2 != (it->second).end() ) {
            (it->second).erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
            return HI_error;
        }
    }

    HI_error_t HI_free_async( const void *hostPtr, int asyncID ) {
        //fprintf(stderr, "[in HI_free_async()] with asyncID %d\n", asyncID);
        postponedFreeTable.insert(std::pair<int, const void *>(asyncID, hostPtr));
        return HI_success;
    }

    HI_error_t HI_postponed_free(int asyncID ) {
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[enter HI_postponed_free()]\n");
#endif
        std::multimap<int, const void*>::iterator hostPtrIter = postponedFreeTable.find(asyncID);

        while(hostPtrIter != postponedFreeTable.end()) {
            //fprintf(stderr, "[in HI_postponed_free()] Freeing on stream %d, address %x\n", asyncID, hostPtrIter->second);
            HI_free(hostPtrIter->second, asyncID);
            hostPtrIter++;
        }

        postponedFreeTable.erase(asyncID);
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[exit HI_postponed_free()]\n");
#endif
        return HI_success;
    }

    HI_error_t HI_get_temphost_address(const void *hostPtr, void **temphostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            *temphostPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = tempHostAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
                *temphostPtr = it2->second;
                return  HI_success;
            }
            //fprintf(stderr, "[ERROR in get_temphost_address()] No mapping found for the host pointer\n");
            return HI_error;
        }
    }

    HI_error_t HI_set_temphost_address(const void *hostPtr, void * temphostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
        //fprintf(stderr, "[in set_temphost_address()] Setting address\n");
        if(it == tempHostAddressTable.end() ) {
            //fprintf(stderr, "[in set_temphost_address()] No mapping found for the asyncID\n");
            std::map<const void *,void*> emptyMap;
            tempHostAddressTable.insert(std::pair<int, std::map<const void *,void*> > (asyncID, emptyMap));
            it = tempHostAddressTable.find(asyncID);
        }

        //(it->second).insert(std::pair<const void *,void*>(hostPtr, temphostPtr));
        (it->second)[hostPtr] = temphostPtr;
        return  HI_success;
    }

    HI_error_t HI_remove_temphost_address(const void *hostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
		if( it != tempHostAddressTable.end() ) {
        	std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        	if(it2 != (it->second).end() ) {
            	(it->second).erase(it2);
            	return  HI_success;
        	} else {
            	fprintf(stderr, "[ERROR in remove_temphost_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
            	return HI_error;
        	}
		} else {
           fprintf(stderr, "[ERROR in remove_temphost_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
           return HI_error;
		}
    }

    void HI_free_temphosts(int asyncID ) {
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[enter HI_free_temphosts()]\n");
#endif
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
		if (it != tempHostAddressTable.end()) {
			for( std::map<const void*,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2 ) {
				HI_tempFree(&(it2->second), acc_device_host);
			}
			(it->second).clear();
		}
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[exit HI_free_temphosts()]\n");
#endif
    }

} Accelerator_t;




#ifdef NVIDIA_GPU
typedef std::map<Accelerator *, std::map<std::string, CUfunction> > kernelmapcuda_t;
#endif

#ifdef NVIDIA_GPU
typedef class CudaDriver: public Accelerator
{
private:
    std::map<int,  CUstream> queueMap;
    std::map<int, eventmap_cuda_t > threadQueueEventMap;

    void pin_host_memory(const void* hostPtr, size_t size);
    void unpin_host_memory(const void* hostPtr);
public:
    static std::vector<std::string> kernelNameList;

    //A map of pinned memory and its usage count. If count value is 0, then the runtime can unpin the host memory.
    static std::map<CUdeviceptr,int> pinnedHostMemCounter;

    //static std::map<std::string, CUfunction> kernelMap;
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;

    CudaDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices);
    HI_error_t init();
    HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
    HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
    HI_error_t HI_synchronize();
    HI_error_t destroy();
    HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
    HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
    HI_error_t HI_free( const void *hostPtr, int asyncID);
    HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
    HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);
    HI_error_t HI_memcpy2D_asyncS(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);

    void HI_tempFree( void** tempPtr, acc_device_t devType);
    void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);

    static int HI_get_num_devices(acc_device_t devType);
    void HI_malloc(void **devPtr, size_t size);
    void HI_free(void *devPtr);
    HI_error_t createKernelArgMap();
    HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size);
    HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count);
    void HI_set_async(int asyncId);
    void HI_async_wait(int asyncId);
    void HI_async_waitS1(int asyncId);
    void HI_async_waitS2(int asyncId);
    void HI_async_wait_all();
    int HI_async_test(int asyncId);
    int HI_async_test_all();
    CUstream getQueue(int async) {
		if( queueMap.count(async + 2) == 0 ) {
			fprintf(stderr, "[ERROR in getQueue()] queue does not exist for async ID = %d\n", async);
			exit(1);
		} 
        return queueMap.at(async + 2);
    }

    CUevent getEvent(int async) {
		int thread_id = get_thread_id();
		if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(async + 2) == 0) ) {
			fprintf(stderr, "[ERROR in getEvent()] event does not exist for async ID = %d and thread ID = %d\n", async, thread_id);
			exit(1);
		}
        return threadQueueEventMap.at(get_thread_id()).at(async + 2);
    }

} CudaDriver_t;
#endif

typedef class OpenCLDriver: public Accelerator
{
private:
    std::map<int,  cl_command_queue> queueMap;
    std::map<int, eventmap_opencl_t > threadQueueEventMap;

public:
    static std::vector<std::string> kernelNameList;
    cl_platform_id clPlatform;
    cl_device_id clDevice;
    cl_context clContext;
    cl_command_queue clQueue;
    cl_program clProgram;

    OpenCLDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices);
    HI_error_t init();
    HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
    HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
    HI_error_t HI_synchronize();
    HI_error_t destroy();
    HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
    HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
    HI_error_t HI_free( const void *hostPtr, int asyncID);
    HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
    HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);

    void HI_tempFree( void** tempPtr, acc_device_t devType);
    void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);

    static int HI_get_num_devices(acc_device_t devType);
    void HI_malloc(void **devPtr, size_t size);
    void HI_free(void *devPtr);
    HI_error_t createKernelArgMap();

    void HI_set_async(int asyncId);
    void HI_async_wait(int asyncId);
    void HI_async_waitS1(int asyncId);
    void HI_async_waitS2(int asyncId);
    void HI_async_wait_all();
    int HI_async_test(int asyncId);
    int HI_async_test_all();

    cl_command_queue getQueue(int async) {
		if( queueMap.count(async + 2) == 0 ) {
			fprintf(stderr, "[ERROR in getQueue()] queue does not exist for async = %d\n", async);
			exit(1);
		} 
        return queueMap.at(async + 2);
    }

    cl_event * getEvent(int async) {
		int thread_id = get_thread_id();
		if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(async + 2) == 0) ) {
			fprintf(stderr, "[ERROR in getEvent()] event does not exist for async ID = %d and thread ID = %d\n", async, thread_id);
			exit(1);
		}
        return &(threadQueueEventMap.at(get_thread_id()).at(async + 2));
    }


} OpenCLDriver_t;

typedef std::map<int, Accelerator_t*> devnummap_t;
typedef std::map<acc_device_t, devnummap_t> devmap_t;
typedef std::map<Accelerator *, std::map<std::string, argmap_t*> > kernelargsmap_t;
typedef std::map<Accelerator *, std::map<std::string, cl_kernel> > kernelmap_t;

class HostConf
{
public:
    Accelerator_t *device;
    kernelargsmap_t kernelArgsMap;
    kernelmap_t kernelsMap;
#ifdef NVIDIA_GPU
    kernelmapcuda_t kernelsMapCUDA;
#endif
    std::vector<std::string> kernelnames;
    static devmap_t devMap;
    HostConf() {
        HI_init_done = 0;
        acc_device_type_var = acc_device_none;
        acc_device_num_var = 0;
        acc_num_devices = 0;
        isOnAccDevice = 0;
#ifdef _OPENARC_PROFILE_
        H2DMemTrCnt = 0;
        H2HMemTrCnt = 0;
        D2HMemTrCnt = 0;
        D2DMemTrCnt = 0;
        HMallocCnt = 0;
        DMallocCnt = 0;
        HFreeCnt = 0;
        DFreeCnt = 0;
        H2DMemTrSize = 0;
        H2HMemTrSize = 0;
        D2HMemTrSize = 0;
        D2DMemTrSize = 0;
#endif
        setDefaultDevice();
        setDefaultDevNum();
    }

    ~HostConf() {
        HI_reset();
        delete device;
    }

    int HI_init_done;
    acc_device_t acc_device_type_var;
    acc_device_t user_set_device_type_var;
    int acc_device_num_var;
    int acc_num_devices;
    int isOnAccDevice;

#ifdef _OPENARC_PROFILE_
    long H2DMemTrCnt;
    long H2HMemTrCnt;
    long D2HMemTrCnt;
    long D2DMemTrCnt;
    long HMallocCnt;
    long DMallocCnt;
    long HFreeCnt;
    long DFreeCnt;
    unsigned long H2DMemTrSize;
    unsigned long H2HMemTrSize;
    unsigned long D2HMemTrSize;
    unsigned long D2DMemTrSize;
    double start_time;
    double end_time;
    double totalWaitTime;
    double totalResultCompTime;
    double totalMemTrTime;
    double totalMallocTime;
    double totalFreeTime;
    double totalACCTime;
    double totalInitTime;
    double totalShutdownTime;
#endif


    memstatusmap_t *hostmemstatusmaptable;
    memstatusmap_t *devicememstatusmaptable;
    countermap_t  *prtcntmaptable;

    void HI_init();
    void HI_reset();
    void setDefaultDevice();
    void setDefaultDevNum();
    void initKernelNames(int kernels, std::string kernelNames[]);

    int genOCL;
    void setTranslationType();
    void createHostTables();

};



extern std::vector<HostConf_t *> hostConfList;

extern int HI_hostinit_done;

extern int HI_openarcrt_verbosity;

extern int HI_num_kernels;

extern std::string *HI_kernelNames;

//////////////////////////
// Moved from openacc.h //
//////////////////////////
extern void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[]);

////////////////////////
// Runtime init/reset //
////////////////////////
extern void HI_hostinit(int numhostthreads);
extern HostConf_t * getInitHostConf();
extern HostConf_t * getHostConf();

//////////////////////
// Kernel Execution //
//////////////////////
extern HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
extern HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
extern HI_error_t HI_synchronize();

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
extern HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID);
extern HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
extern HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
extern HI_error_t HI_free( const void *hostPtr, int asyncID);
extern HI_error_t HI_free_async( const void *hostPtr, int asyncID);
extern void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);
extern void HI_tempFree( void** tempPtr, acc_device_t devType);


/////////////////////////////////////////////////
//Memory transfers between a host and a device //
/////////////////////////////////////////////////
extern HI_error_t HI_memcpy(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType);
extern HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async);
extern HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async);
extern HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                                    size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
extern HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);
//extern HI_error_t HI_memcpy3D(void *dst, size_t dpitch, const void *src, size_t spitch,
//	size_t widthInBytes, size_t height, size_t depth, HI_MemcpyKind_t kind);
//extern HI_error_t HI_memcpy3D_async(void *dst, size_t dpitch, const void *src,
//	size_t spitch, size_t widthInBytes, size_t height, size_t depth,
//	HI_MemcpyKind_t kind, int async);
extern HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count);

////////////////////////////
//Internal mapping tables //
////////////////////////////
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, int asyncID);
extern HI_error_t HI_set_device_address(const void * hostPtr, void * devPtr, int asyncID);
extern HI_error_t HI_remove_device_address(const void * hostPtr);
extern HI_error_t HI_get_temphost_address(const void * hostPtr, void ** temphostPtr, int asyncID);
extern HI_error_t HI_set_temphost_address(const void * hostPtr, void * temphostPtr, int asyncID);
extern HI_error_t HI_remove_temphost_address(const void * hostPtr);
extern int HI_getninc_prtcounter(const void * hostPtr, void **devPtr, int asyncID);
extern int HI_decnget_prtcounter(const void * hostPtr, void **devPtr, int asyncID);

/////////////////////////////////////////////////////////////////////////
//async integer argument => internal handler (ex: CUDA stream) mapping //
/////////////////////////////////////////////////////////////////////////
extern HI_error_t HI_create_async_handle( int async);
extern int HI_contain_async_handle( int async );
extern HI_error_t HI_delete_async_handle( int async );

#ifdef NVIDIA_GPU
extern HI_async_handle_t HI_get_async_handle( int async );
extern HI_error_t HI_set_async_handle( int async, HI_async_handle_t handler );
#endif

extern void HI_set_async(int asyncId);
////////////////////////////////
//Memory management functions //
////////////////////////////////
extern void HI_check_read(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex);
extern void HI_check_write(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex);
extern void HI_set_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, const char * varName, const char * refName, int loopIndex);
extern void HI_reset_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, int asyncID);
//Below is deprecated
extern void HI_init_status(const void * hostPtr);

////////////////////
//Texture function //
////////////////////
extern HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size);

////////////////////
//Misc. functions //
////////////////////
extern double HI_get_localtime();


////////////////////////////////////////////
//Functions used for program verification //
////////////////////////////////////////////
extern void HI_async_waitS1(int asyncId);
extern void HI_async_waitS2(int asyncId);


///////////////////////////////////////
//Functions used for resilience test //
///////////////////////////////////////
#include "resilience.h"



#endif



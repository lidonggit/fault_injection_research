#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include "openaccrt.h"

static const char *openarcrt_verbosity_env = "OPENARCRT_VERBOSITY";

int HI_num_kernels;
std::string *HI_kernelNames;

int get_thread_id() {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    return thread_id;
}

int acc_get_num_devices( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_num_devices()\n");
	}
#endif
    HostConf *tconf = getHostConf();
    int count;
    tconf->setTranslationType();

    if( HI_hostinit_done == 0 ) {
        HI_hostinit(0);
    }
    //DEBUG: Do we need to set up CUDA context?
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon)) {
        devtype = acc_device_gpu;
        if(tconf->genOCL) {
            count = OpenCLDriver::HI_get_num_devices(devtype);
        } else {
#ifdef NVIDIA_GPU
            count = CudaDriver::HI_get_num_devices(devtype);
#endif
        }
    } else if( devtype == acc_device_host ) {
        count = 1;
    } else {
        count = 0;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_num_devices()\n");
	}
#endif
    return count;
}

void acc_set_device_type( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_type()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->user_set_device_type_var =  devtype;
    if( devtype == acc_device_default || devtype == acc_device_nvidia || devtype == acc_device_radeon ) {
        tconf->user_set_device_type_var = acc_device_gpu;
        tconf->acc_device_type_var = acc_device_gpu;
    } else if ( devtype == acc_device_not_host) {
        tconf->acc_device_type_var = acc_device_gpu;
    } else {
        tconf->acc_device_type_var = devtype;
    }

    tconf->setDefaultDevNum();
    acc_set_device_num(tconf->acc_device_num_var, tconf->user_set_device_type_var);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_type()\n");
	}
#endif
}

acc_device_t acc_get_device_type(void) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_type()\n");
	}
#endif
	acc_device_t return_data;
    HostConf_t * tconf = getHostConf();
    return_data = tconf->user_set_device_type_var;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_type()\n");
	}
#endif
	return return_data;
}

//If the value of devicenum is negative, the runtime will revert to its
//default behavior, which is implementation-defined. If the value
//of the second argument is zero, the selected device number will
//be used for all attached accelerator types.
void acc_set_device_num( int devnum, acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_num()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->user_set_device_type_var = devtype;
	if( devnum < 0 ) {
		devnum = 0;
	}
    if( devtype == acc_device_nvidia ||  devtype == acc_device_radeon ||  devtype == acc_device_gpu ) {
        //tconf->acc_device_num_var = devnum;
        devtype = acc_device_gpu;
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
			fprintf(stderr, "Host Thread %d uses device %d\n",get_thread_id(), devnum);
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        tconf->acc_device_type_var = acc_device_gpu;
        tconf->user_set_device_type_var =acc_device_gpu;
        tconf->acc_device_num_var = devnum;
        //printf("devType %d\n",devtype );
        if(tconf->device->init_done != 1) {
            tconf->device->init();
        } else {
            tconf->device->createKernelArgMap();
        }
        //printf("kernel arg map created\n");
    } else if( devtype == acc_device_xeonphi ) {
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
			fprintf(stderr, "Host Thread %d uses device %d\n",get_thread_id(), devnum);
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        tconf->acc_device_type_var = acc_device_xeonphi;
        tconf->acc_device_num_var = devnum;
        //printf("devType %d\n",devtype );
        if(tconf->device->init_done != 1) {
            tconf->device->init();
        } else {
            tconf->device->createKernelArgMap();
        }
        //printf("kernel arg map created\n");
    } else if (devtype == acc_device_not_host) {
        tconf->setDefaultDevice();
		int numDevs = HostConf::devMap.at(tconf->acc_device_type_var).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
			fprintf(stderr, "Host Thread %d uses device %d\n",get_thread_id(), devnum);
		}
        tconf->device = HostConf::devMap.at(tconf->acc_device_type_var).at(devnum);
        tconf->acc_device_num_var = devnum;
        if(tconf->device->init_done != 1) {
            tconf->device->init();
        } else {
            tconf->device->createKernelArgMap();
        }

    } else if (devtype == acc_device_default) {
        tconf->setDefaultDevice();
		int numDevs = HostConf::devMap.at(tconf->acc_device_type_var).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
			fprintf(stderr, "Host Thread %d uses device %d\n",get_thread_id(), devnum);
		}
        tconf->device = HostConf::devMap.at(tconf->acc_device_type_var).at(devnum);
        tconf->acc_device_num_var = devnum;
        if(tconf->device->init_done != 1) {
            tconf->device->init();
        } else {
            tconf->device->createKernelArgMap();
        }
        tconf->user_set_device_type_var = acc_device_gpu;
    } else if( devtype == acc_device_host ) {
        tconf->acc_device_num_var = devnum;
        //tconf->device = tconf->devMap.at(devtype).at(devnum);
    } else {
        fprintf(stderr, "[ERROR in acc_set_device_num()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
    /*
    if( tconf->isOnAccDevice > 0 ) {

    	CUDA device number starts from 0, but OpenACC device number starts
    	from 1. (0 is for default device in OpenACC.)
    	From OpenACC V2.0, device number starts from 0; we will follow the
    	new rule.

    	if( devnum >= 0 ) {
    		cudaSetDevice(devnum);
    	} else {
    		fprintf(stderr, "[ERROR in acc_set_device_num()] Not supported device number: %d; exit!\n", devnum);
    		exit(1);
    	}
    }*/
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_num()\n");
	}
#endif
}

int acc_get_device_num( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_num()\n");
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf();
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) || (devtype == acc_device_gpu)  ) {
        return_data = tconf->acc_device_num_var;
    } else if( devtype == acc_device_host ) {
        return_data = tconf->acc_device_num_var;
    } else {
        fprintf(stderr, "[ERROR in acc_get_device_num()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_num()\n");
	}
#endif
	return return_data;
}

int acc_async_test( int asyncID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test(%d)\n", asyncID);
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf();
    return_data = tconf->device->HI_async_test(asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test(%d)\n", asyncID);
	}
#endif
	return return_data;
}

int acc_async_test_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test_all()\n");
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf();
    return_data = tconf->device->HI_async_test_all();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test_all()\n");
	}
#endif
	return return_data;
}

void acc_async_wait( int asyncID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_wait(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_async_wait(asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_wait(%d)\n", asyncID);
	}
#endif
}

void acc_async_wait_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_wait_all()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_async_wait_all();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_wait_all()\n");
	}
#endif
}


void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[] ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_init()\n");
	}
#endif
	HI_num_kernels = kernels;
	HI_kernelNames = kernelNames;
    HostConf_t * tconf = getInitHostConf();
    //Set device type.
    if( devtype != acc_device_default ) {
        tconf->acc_device_type_var = devtype;
    }
    //tconf->HostConf::HI_specify_kernel_names(kernels, kernelNames, 0);
    tconf->initKernelNames(kernels, kernelNames);
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1;
    	tconf->HI_init();
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_init()\n");
	}
#endif
}

void acc_shutdown( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_shutdown()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    if( tconf == NULL ) {
        return;
    }
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) 
            || (devtype == acc_device_xeonphi)) {
        if( tconf->device->init_done == 1 ) {
            fflush(stdout);
            fflush(stderr);
            tconf->isOnAccDevice = 0;

            //[DEBUG] below statements are moved into HI_reset()
            //fprintf(stderr, "[in acc_shutdown()] about to destroy!\n");
            //tconf->device->masterAddressTable.clear();
            //tconf->device->postponedFreeTable.clear();
            //tconf->device->destroy();
            //fprintf(stderr, "[in acc_shutdown()] destroy done!\n");
            //tconf->device->init_done = 0;
            //fprintf(stderr, "[in acc_shutdown()] about to reset\n");
            tconf->HI_reset();
            //fprintf(stderr, "[in acc_shutdown()] reset done!\n");
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_shutdown()\n");
	}
#endif
}

//DEBUG: below implementation can be called only by host threads.
//Call to this function within a GPU kernel should be overwritten
//by OpenACC-to-CUDA translator.
int acc_on_device( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_on_device()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    //TODO:
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon)
            || (devtype == acc_device_xeonphi)) {
        //return tconf->isOnAccDevice;
        return 0;
    } else if( devtype == acc_device_host ) {
        //return tconf->isOnAccDevice == 0 ? 1 : 0;
        return 1;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_on_device()\n");
	}
#endif
    return 0;
}

void* acc_malloc(size_t size) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_malloc()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    void *devPtr;
    if( tconf->isOnAccDevice ) {
        tconf->device->HI_malloc(&devPtr, size);
    } else {
        fprintf(stderr, "[ERROR in acc_malloc()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_malloc()\n");
	}
#endif
    return devPtr;
}

void acc_free(void* devPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_free()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    if( tconf->isOnAccDevice ) {
        tconf->device->HI_free(devPtr);
    } else {
        fprintf(stderr, "[ERROR in acc_free()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_free()\n");
	}
#endif
}


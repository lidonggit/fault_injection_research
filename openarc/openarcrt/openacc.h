#ifndef __OPENACC_HEADER__

#define __OPENACC_HEADER__

//#include <string>
#include <stddef.h>

typedef enum {
    acc_device_none = 0,
    acc_device_default = 1,
    acc_device_host = 2,
    acc_device_not_host = 3,
    acc_device_nvidia = 4,
    acc_device_radeon = 5,
    acc_device_gpu = 6,
    acc_device_xeonphi = 7,
    acc_device_current = 8
} acc_device_t;

extern int acc_get_num_devices( acc_device_t devtype );
extern void acc_set_device_type( acc_device_t devtype );
extern acc_device_t acc_get_device_type(void);
extern void acc_set_device_num( int devnum, acc_device_t devtype );
extern int acc_get_device_num( acc_device_t devtype );
extern int acc_async_test( int asyncID );
extern int acc_async_test_all();
extern void acc_async_wait( int asyncID );
extern void acc_async_wait_all();
extern void acc_init( acc_device_t devtype );
extern void acc_shutdown( acc_device_t devtype );
extern int acc_on_device( acc_device_t devtype );
extern void* acc_malloc(size_t);
extern void acc_free(void*);
extern int get_thread_id();
//Below will be added separately.
//#include "openaccrt.h"

#endif

#ifndef _LIBFI
#define _LIBFI


#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG

void launch_fi_thread(void* start_address, int mem_size);

void launch_fi_thread_(void *start_address, int *size);


#ifdef __cplusplus
}
#endif

#endif //_LIBFI

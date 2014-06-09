#ifndef _LIBFI
#define _LIBFI


#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG

void launch_fi_thread(void*,int,double,double,double,pthread_t);

void launch_fi_thread_(void*,int*,double*,double*,double*,pthread_t*);


#ifdef __cplusplus
}
#endif

#endif //_LIBFI

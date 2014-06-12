#ifndef _LIBFI
#define _LIBFI

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG

typedef void (*sighandler_t)(int);

void launch_fi_thread(void*,int,double,double,double);

void launch_fi_thread_(void*,int*,double*,double*,double*);

void show_stackframe();

void sig_handler(int);

void sig_handler_(int*);

#ifdef __cplusplus
}
#endif

#endif //_LIBFI

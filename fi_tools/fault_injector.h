#ifndef _LIBFI
#define _LIBFI

#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG

void launch_fi_thread(void*,int,double,double,double);

void launch_fi_thread_(void*,int*,double*,double*,double*);

void sig_handler(int);

void sig_handler_(int*);

void show_stackframe();

void bit_flip();

#ifdef __cplusplus
}
#endif

#endif //_LIBFI

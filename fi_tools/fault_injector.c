#include <execinfo.h>
#include <fault_injector.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/// some global variables
int mem_size;
double exe_time;
double time_random;
double space_random;

/// printing out stack trace
void show_stackframe() {
  void *trace[16];
  char **messages = (char **)NULL;
  int i, trace_size = 0;
  trace_size = backtrace(trace, 16);
  messages = backtrace_symbols(trace, trace_size);
  printf("[bt] Execution path:\n");
  for (i=0; i<trace_size; ++i)
  {
      printf("[bt] %s\n", messages[i]);
  }
}

void sig_handler(int sig)
{
//  write(1, "Caught signal from fault injector ... \n", 17);
//  signal(sig,sig_func);
    show_stackframe();
}

void *fault_injection(void *start_address)
{
    ///get the start time
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);

    ///generate randomness
//    srand((unsigned)time(NULL));
//    float time_random = random_generator();
//    float space_random = random_generator();

    #ifdef DEBUG
    printf("time_random=%f, space_random=%f, exe_time=%f seconds, mem_size=%d bytes\n", time_random, space_random, exe_time, mem_size);
    #endif

    ///decide the time to trigger fault injection
    unsigned int trigger_time, waittime;
    trigger_time = (exe_time * 1000000) * time_random;  ///in microseconds
    gettimeofday(&current_time, NULL);
    waittime = trigger_time -
               ((current_time.tv_sec*1000000 + current_time.tv_usec)-
               (start_time.tv_sec*1000000 + start_time.tv_usec)) ;

    if(waittime <= 0)
    {
        printf("Waittime is no bigger than 0. Will perform fault injection immediately\n");
    }
    else
    {
        //#ifdef DEBUG
        printf("*****[fault_injection tool] start_addr=%p, mem_size=%d bytes, tigger_time=%d us, waittime=%d us*****\n",
               start_address, mem_size, trigger_time, waittime);
        //#endif
        usleep(waittime);
    }

    ///perform random fault injection
    int fi_bit_point, fi_byte_point;
    fi_bit_point = (int)(mem_size * 8 * space_random); ///a random bit
    fi_byte_point = fi_bit_point/8;
    char * target_byte = (char *)start_address + fi_byte_point;  ///char* is 1 byte
    ///#ifdef DEBUG
    printf("*****[fault_injection tool] mem_size*8*space_random=%f, fi_bit_point=%d, fi_byte_point=%d, value_of_target_byte(before fi)=%x *****\n",
	       mem_size*8*space_random, fi_bit_point, fi_byte_point, *target_byte);
    ///#endif

    *target_byte ^= (1UL << (fi_bit_point - fi_byte_point*8));

//    printf("tid %u\n", (unsigned int)mtid);
//    pthread_kill(mtid, 12);

    /// sending signal to the process
    char command[30] = "kill -s 12 ";
    char pid[15];
    sprintf(pid, "%d", (int)getpid());
    strcat(command,pid);
    if(popen(command,"r") == NULL)
    {
        printf("Command sending fail: %s\n",command);
        exit(-1);
    }

    ///#ifdef DEBUG
    printf("*****[fault_injection tool] value_of_target_byte(after fi)=%x, address_for_target_byte=%p *****\n",
           *target_byte, (void *)target_byte);
    ///#endif

    pthread_exit(NULL);
}

//This API needs to be inserted right before the computation starts
void launch_fi_thread(void* start_address, int size, double time, double time_rand, double space_rand)
{
    int rc;
    pthread_t fi_thread;
    mem_size = size;
    exe_time = time;
    time_random = time_rand;
    space_random = space_rand;

    #ifdef DEBUG
    printf("start_address=%p\n", start_address);
    #endif

    rc = pthread_create(&fi_thread, NULL, fault_injection, start_address);

    if(rc)
    {
        printf("Error: return code from pthread_create() is %d. Pthread creation fails\n", rc);
        exit(-1);
    }
}

//This API is for Fortran
void launch_fi_thread_(void *start_address, int *size, double *time, double *time_rand, double *space_rand)
{
    launch_fi_thread(start_address, *size, *time, *time_rand, *space_rand);
}

void sig_handler_(int *sig)
{
    show_stackframe();
}

/*
int pthread_self_()
{
    int tid = pthread_self();
    printf("tid %u\n", (unsigned int)tid);
    return (tid);
}

void signal_( int* signum, sighandler_t handler)
{
    signal(*signum, handler);
}

void sigclear_(int *signum)
{
    signal(*signum, NULL);
}
*/

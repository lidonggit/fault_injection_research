#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fault_injector.h>

int mem_size;
double exe_time;

double random_generator()
{
    return (double)rand()/RAND_MAX;
}

void *fault_injection(void *start_address)
{
    ///get the start time
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);

    ///generate randomness
    srand((unsigned)time(NULL));
    float time_random = random_generator();
    float space_random = random_generator();

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

    ///#ifdef DEBUG
    printf("*****[fault_injection tool] value_of_target_byte(after fi)=%x, address_for_target_byte=%p *****\n",
           *target_byte, (void *)target_byte);
    ///#endif

    pthread_exit(NULL);
}

//This API needs to be inserted right before the computation starts
void launch_fi_thread(void* start_address, int size, double time)
{
    int rc;
    pthread_t fi_thread;
    mem_size = size;
    exe_time = time;

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
void launch_fi_thread_(void *start_address, int *size, double *time)
{
    launch_fi_thread(start_address, *size, *time);
}

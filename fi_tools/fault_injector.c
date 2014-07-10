#include <execinfo.h>
#include <fault_injector.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

/// global variables
struct timeval start_time, current_time;
struct timeval sendtime, recvtime;
void* start_address;
int mem_size;
double exe_time;
double time_random, rtime_random;
double space_random;

void bit_flip()
{
    printf("Signal Delay: %dus\n",(int)recvtime.tv_usec-(int)sendtime.tv_usec);
    unsigned int fi_bit_point, fi_byte_point;
    fi_bit_point = (unsigned int)((float)mem_size*8*space_random);
    fi_byte_point = fi_bit_point/8;
    char* target_byte = (char *)start_address+fi_byte_point;
    #ifdef DEBUG
    printf("[Before FI] fi_bit_point=%u, fi_byte_point=%u, value_of_target_byte=%x\n",         
           fi_bit_point, fi_byte_point, *target_byte);
    #endif
    *target_byte ^= (1UL<<(fi_bit_point-fi_byte_point*8));
    #ifdef DEBUG
    printf("[After  FI] address_for_target_byte=%p, value_of_target_byte=%x\n",
           (void *)target_byte, *target_byte);
    #endif

    unsigned int elapse_time = (recvtime.tv_sec*1000000+recvtime.tv_usec)- 
                               (start_time.tv_sec*1000000+start_time.tv_usec);
    rtime_random = (float)elapse_time / (exe_time * 1000000);
    printf("real_time_random=%f\n",rtime_random);
}

void show_stackframe()
{
    int child_pid = fork();
    if(!child_pid) 
    {
        int i, idx=0;
        char pid[30];
        char pname[256];
        char fname[20];
//      char gdbpath[] = "/home/li/Documents/lib/bin/gdb";
        char gdbpath[] = "/home/lyu17/libs/gdb/bin/gdb";
        sprintf(pid,"%d",getppid());
        pname[readlink("/proc/self/exe",pname,255)]=0;
        for(i=0;i<strlen(pname);i++)
        {
            if(pname[i]=='/')
            {
                idx = i;
            }
        }
        strncpy(fname,pname+idx+1,20);
        strcat (fname,".txt");
        int fd = open(fname,O_WRONLY|O_CREAT|O_APPEND,S_IRUSR|S_IWUSR);
        dup2(fd,1); 
        dup2(fd,2); 
        close(fd);
        fprintf(stderr,"time_random=%f space_random=%f\n",time_random,space_random);
        fprintf(stderr,"stack trace for %s pid=%s\n",pname,pid);
        execlp(gdbpath,gdbpath,"--batch","-n","-ex","thread","-ex","bt",pname,pid,NULL);
        fprintf(stderr,"\n\n");
        abort();
    }
    else 
    {
        waitpid(child_pid,NULL,0);
    }
}

void sig_handler(int sig)
{
    gettimeofday(&recvtime, NULL);
    show_stackframe();
    bit_flip();
}

void *fault_injection()
{
    ///get the start time
    gettimeofday(&start_time, NULL);

    #ifdef DEBUG
    printf("[FI] exec_time=%f, time_random=%f, mem_size=%d bytes, space_random=%f\n",     
           exe_time, time_random, mem_size, space_random);
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
        #ifdef DEBUG
        printf("[FI] start_addr=%p, mem_size=%d bytes, tigger_time=%d us, waittime=%d us\n",  
               start_address, mem_size, trigger_time, waittime);
        #endif
        usleep(waittime);
    }

    /// sending signal to the process
    char command[30] = "kill -s 12 ";
    char pid[15];
    sprintf(pid, "%d", (int)getpid());
    strcat(command,pid);
    FILE *ptr = NULL;

    gettimeofday(&sendtime, NULL);

    if((ptr=popen(command,"r"))==NULL)
    {
        printf("Command Sending Fail: %s\n",command);
        exit(-1);
    }

    pthread_exit(NULL);
}

//This API needs to be inserted right before the computation starts

void launch_fi_thread(void* address, int size, double time, double time_rand, double space_rand)
{
    int rc;
    pthread_t fi_thread;
    start_address = address;
    mem_size = size;
    exe_time = time;
    time_random = time_rand;
    space_random = space_rand;

    #ifdef DEBUG
    printf("start_address=%p\n", start_address);
    #endif

    rc = pthread_create(&fi_thread, NULL, fault_injection, NULL);

    if(rc)
    {
        printf("Error: return code from pthread_create() is %d. Pthread creation fails\n", rc);
        exit(-1);
    }

}

//This API is for Fortran

void launch_fi_thread_(void *address, int *size, double *time, double *time_rand, double *space_rand)
{
    launch_fi_thread(address, *size, *time, *time_rand, *space_rand);
}

void sig_handler_(int *sig)
{
    gettimeofday(&recvtime, NULL);
    show_stackframe();
    bit_flip();
}

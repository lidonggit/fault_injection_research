#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fault_injector.h>

int mem_size;

void *fault_injection(void *start_address)
{
   //get the start time;
   //http://www.cs.loyola.edu/~jglenn/702/S2008/Projects/P3/time.html
   //http://man7.org/linux/man-pages/man2/settimeofday.2.html
   struct timeval start_time, current_time;
   gettimeofday(&start_time, NULL);
	
   //read the configuration file
   FILE *fp;
   char * config_path;
   config_path = getenv("FI_CONFIG_PATH");
   if(config_path != NULL)
   {
#ifdef DEBUG
     printf("FI_CONFIG_PATH=%s\n", config_path);
#endif
     fp = fopen(config_path, "r"); 
   }
   else
     fp = fopen("./config_file.cfg", "r");

   if(fp == NULL){
      printf("Error: can't open the input configuration file."
		" The file path is %s\n", config_path);
      exit(1);
   }   
 	
   float time_random, space_random, exe_time;
   fscanf(fp, "%f", &time_random);
   fscanf(fp, "%f", &space_random);
   fscanf(fp, "%f", &exe_time); //in seconds
   //fscanf(fp, "%d", &mem_size); //in seconds

#ifdef DEBUG
   printf("time_random=%f, space_random=%f, exe_time=%f, mem_size=%d\n", 
	time_random, space_random, exe_time, mem_size);  
#endif

   //decide the time to trigger fault injection
   unsigned int trigger_time, waittime;
   trigger_time = (exe_time * 1000000) * time_random;  //in microseconds
   gettimeofday(&current_time, NULL);
   waittime = trigger_time - 
		((current_time.tv_sec*1000000 + current_time.tv_usec)-
		 (start_time.tv_sec*1000000 + start_time.tv_usec)) ;
   
   if(waittime <= 0)
      printf("waittime is no bigger than 0."
	     "Will perform fault injection immediately\n");
   else
   {
#ifdef DEBUG
     printf("tigger_time=%d, waittime=%d\n", trigger_time, waittime);
#endif
     usleep(waittime);   
   } 

    //perform random fault injection
    int fi_bit_point, fi_byte_point;
    fi_bit_point = (int)(mem_size * 8 * space_random); //a random bit
    fi_byte_point = fi_bit_point/8;
    char * target_byte = (char *)start_address + fi_byte_point;  //char* is 1 byte
#ifdef DEBUG
    printf("mem_size*8*space_random=%f, fi_bit_point=%d, fi_byte_point=%d, target=%d\n", 
	   mem_size*8*space_random, fi_bit_point, fi_byte_point, *target_byte);
#endif

    *target_byte ^= (1UL << (fi_bit_point - fi_byte_point*8));

#ifdef DEBUG
    printf("target(after fi)=%d, target_byte=%p\n", *target_byte, (void *)target_byte);
#endif
   
    pthread_exit(NULL); 
}

//This API needs to be inserted right before the computation starts
void launch_fi_thread(void* start_address, int size)
{
   int rc;
   pthread_t fi_thread;
   mem_size = size;

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



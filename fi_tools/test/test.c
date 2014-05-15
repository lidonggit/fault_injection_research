#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fault_injector.h>

#define DUMMY_SIZE  1000 

int global_sum=0;
float dummy[DUMMY_SIZE];
int func1()
{
   int *p = NULL;
   int func1_sum=0;
   p = (int *)malloc(sizeof(int)*1000);
   int i;
   for(i=0; i<1000; i++)
   {
     p[i] = i;
   }
   for(i=1; i<1000; i++)
   {
     p[i] = p[i-1]*p[i+1];
     func1_sum = p[i] + func1_sum;
   }
   free(p);

   dummy[0]++;
   return func1_sum;
}

int func2()
{
   int *p = NULL;
   int func2_sum=0;
   p = (int *)malloc(sizeof(int)*1000);
   int i;
   for(i=1000; i<2000; i++)
   {
     p[i-1000] = i;
   }
   for(i=1; i<1000; i++)
   {
     p[i] = p[i-1]*p[i+1];
     func2_sum = p[i] + func2_sum;
   }
   free(p);
   dummy[44]++;
   return func2_sum;
}


main(){
  int i;
  printf("sizeof(int)=%d\n", sizeof(int));
  static int test_static=1;
  printf("test_static=%d\n", test_static);


  launch_fi_thread((void *)dummy);
  for(i=0; i<10; i++)
  {
    usleep(3000000);  //3s
    printf("iter=%d\n", i);
    global_sum = func1() + global_sum;
    global_sum = func2() + global_sum;
    dummy[i] = dummy[i] + global_sum;
  }

  printf("global_sum=%d\n", global_sum);
 
}

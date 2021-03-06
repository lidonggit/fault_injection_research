#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#if OMP == 1
#include <omp.h>
#endif

#include <resilience.h>

#ifdef __cplusplus
#define restrict __restrict__
#endif

#ifndef _N_
#define _N_ 512
#endif

#define ITER 10

int N = _N_;
int M = N;
int P = N;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

#pragma acc #define TTHREAD 0
#pragma acc #define FTVAR b[0:(M*P)]
#pragma acc #define TOTAL_NUM_FAULTS 1
#define TOTAL_NUM_FAULTS 1
#pragma acc #define NUM_FAULTYBITS 1


void
MatrixMultiplication_openacc(float * restrict a,float * restrict b, float * restrict c)
{
  int i, j, k, m, l;

  unsigned int itrpos[TOTAL_NUM_FAULTS];
  unsigned int injectFT = 0;

  //Decide at which iteration to inject fault.
  HI_set_srand();
  for( l=0; l<TOTAL_NUM_FAULTS; l++ ) {
    itrpos[l] = HI_genrandom_int(ITER);
  }
  HI_sort_int(itrpos, TOTAL_NUM_FAULTS);

#pragma acc data copyout(a[0:(M*N)]), copyin(b[0:(M*P)],c[0:(P*N)])
  for( m=0; m<ITER; m++) {
    //Enable fault injection only at randomly selected iterations.
    injectFT = 0;
    for (l=0; l<TOTAL_NUM_FAULTS; l++) {
      if( m == itrpos[l] ) {
        injectFT = 1;
      }
    }
#pragma acc resilience ftregion ftthread(TTHREAD) ftcond(injectFT) ftdata(FTVAR) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#pragma acc kernels loop independent gang
    for (i=0; i<M; i++){
#pragma acc loop worker
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
#pragma acc loop seq
	  for (k=0; k<P; k++) {
	    sum += b[i*P+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
  }
}


void
MatrixMultiplication_openmp(float * restrict a,float * restrict b, float * restrict c)
{
  int i, j, k ;
  int chunk = N/4;


#pragma acc resilience ftregion ftdata(FTVAR) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#pragma omp parallel shared(a,b,c,chunk) private(i,j,k)
  {
#ifdef _OPENMP
	if(omp_get_thread_num() == 0) {
		printf("Number of OpenMP threads %d\n", omp_get_num_threads());
	}
#endif
#pragma omp for
    for (i=0; i<M; i++){
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
	  for (k=0; k<P; k++)
	    sum += b[i*P+k]*c[k*N+j] ;
	  a[i*N+j] = sum ;
        }
    }
  }
}


int main()
{
  float *a, *b, *c;
  int i;
  double elapsed_time;

  a = (float *) malloc(M*N*4);
  b = (float *) malloc(M*P*4);
  c = (float *) malloc(P*N*4);

  for (i = 0; i <  M*N; i++) {
    a[i] = (float) 0.0;
  }
  for (i = 0; i <  M*P; i++) {
    b[i] = (float) i;
  }
  for (i = 0; i <  P*N; i++) {
    c[i] = (float) 1.0;
  }

  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
  elapsed_time = my_timer();
  MatrixMultiplication_openacc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

  free(a);
  free(b);
  free(c);

  return 0;
} 


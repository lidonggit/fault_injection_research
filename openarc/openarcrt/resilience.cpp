////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
#include "resilience.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <cstring>

#define _DEBUG_FTPRINT_ON_ 1

typedef std::map<const void *, int> cs_optionmap_t;
typedef std::map<const void *, long> cs_countermap_t;
typedef std::map<const void *, long> cs_intchecksummap_t;
typedef std::map<const void *, double> cs_floatchecksummap_t;
typedef std::map<const void *, void *> cp_checkpointmap_t;

static int HI_srand_set = 0;
//interal maps for checksum
static cs_optionmap_t cs_optionmap;
static cs_countermap_t cs_sizemap;
static cs_intchecksummap_t cs_intchecksummap;
static cs_floatchecksummap_t cs_floatchecksummap;
//interal maps for checkpoint
static cs_optionmap_t cp_optionmap;
static cs_countermap_t cp_sizemap;
static cp_checkpointmap_t cp_checkpointmap;

void HI_set_srand() {
    struct timeval time;
    gettimeofday(&time, 0); 
    unsigned int seed = time.tv_sec*time.tv_usec;
    srand(seed);
	//printf("execute HI_set_srand() with seed %d\n", seed);
	HI_srand_set = 1;
}

type8b HI_genbitvector8b(int numFaults) {
    int j;
    unsigned int bit;
    type8b bitVector = 0;
	type8b tbitvec = 1;
    double numBits = 8.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type16b HI_genbitvector16b(int numFaults) {
    int j;
    unsigned int bit;
    type16b bitVector = 0;
	type16b tbitvec = 1;
    double numBits = 16.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type32b HI_genbitvector32b(int numFaults) {
    int j;
    unsigned int bit;
    type32b bitVector = 0;
	type32b tbitvec = 1;
    double numBits = 32.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type64b HI_genbitvector64b(int numFaults) {
    int j;
    unsigned int bit;
    type64b bitVector = 0;
	type64b tbitvec = 1;
    double numBits = 64.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

unsigned long int HI_genrandom_int(unsigned long int Range) {
    unsigned long int rInt;
    double dRange = (double)Range;
	if( HI_srand_set == 0 ) {
		printf("set srand in HI_genrandom_int()\n");
		HI_set_srand();
	}
    rInt = (unsigned long int)(dRange * rand()/(RAND_MAX + 1.0));
    return rInt;
}

void HI_sort_int( unsigned int* iArray, int iSize ) {
	int i, j;
	unsigned int tmp;
	int middle;
	int left, right;
	for( i=1; i<iSize; ++i ) {
		tmp = iArray[i];
		left = 0;
		right = i;
		while (left < right) {
			middle = (left + right)/2;
			if( tmp >= iArray[middle] ) {
				left = middle + 1;
			} else {
				right = middle;
			}	
		}	
		for ( j=i; j>left; --j ) {
			//swap(j-1, j);
			tmp = iArray[j-1];
			iArray[j-1] = iArray[j];
			iArray[j] = tmp;
		}
	} 
}

void HI_ftinjection_int8b(type8b * target, int ftinject,  long int epos, type8b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int8b data\n");
#endif
    }   
}

void HI_ftinjection_int16b(type16b * target, int ftinject,  long int epos, type16b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int16b data\n");
#endif
    }   
}

void HI_ftinjection_int32b(type32b * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int32b data\n");
#endif
    }   
}

void HI_ftinjection_int64b(type64b * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int64b data\n");
#endif
    }   
}

/*
void HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        type32b val = (type32b)(*(target+epos));
        val ^= bitvec;
        *(target+epos) = (float)val;
    }   
}

void HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        type64b val = (type64b)(*(target+epos));
        val ^= bitvec;
        *(target+epos) = (double)val;
    }   
}
*/

void HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
		FloatBits val;
        val.f = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.f;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for float data\n");
#endif
    }   
}

void HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
		DoubleBits val;
        val.d = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.d;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for double data\n");
#endif
    }   
}

void HI_checksum_register_intT(const void * target, long int size, int option) {
	cs_sizemap[target] = size;
	cs_optionmap[target] = option;
	if( option == 0 ) {
		cs_intchecksummap[target] = 0; //set initial value	
	}
}

void HI_checksum_register_floatT(const void * target, long int size, int option) {
	cs_sizemap[target] = size;
	cs_optionmap[target] = option;
	if( option == 0 ) {
		cs_floatchecksummap[target] = 0.0; //set initial value	
	}
}

template<typename T>
void HI_checksum_set_intT(T target) {
	long int size;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_set_intT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			long int checksum = 0;
			for( i=0; i<size; i++ ) {
				checksum += (long int)(*(target+i));
			}
			cs_intchecksummap[target] = checksum;
		}
	}
}

template<typename T>
void HI_checksum_set_floatT(T target) {
	long int size;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_set_floatT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			double checksum = 0;
			for( i=0; i<size; i++ ) {
				checksum += double(*(target+i));
			}
			cs_floatchecksummap[target] = checksum;
		}
	}
}

template<typename T>
int HI_checksum_check_intT(T target) {
	int error = 0;
	long int size;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ||
		(cs_intchecksummap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_check_intT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			long int checksum = 0;
			long int checksum_old = 0;
			checksum_old = cs_intchecksummap[target];
			for( i=0; i<size; i++ ) {
				checksum += (long int)(*(target+i));
			}
			if( checksum != checksum_old ) {
				error = 1;
			}
			cs_intchecksummap[target] = checksum;
		}
	}
	return error;
}

template<typename T>
int HI_checksum_check_floatT(T target) {
	int error = 0;
	long int size;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ||
		(cs_floatchecksummap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_check_floatT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			double checksum = 0;
			double checksum_old = 0;
			checksum_old = cs_floatchecksummap[target];
			for( i=0; i<size; i++ ) {
				checksum += double(*(target+i));
			}
			if( checksum != checksum_old ) {
				error = 1;
			}
			cs_floatchecksummap[target] = checksum;
		}
	}
	return error;
}


void HI_checksum_set_int8b(type8b * target) {
	HI_checksum_set_intT<type8b *>(target);	
}

void HI_checksum_set_int16b(type16b * target) {
	HI_checksum_set_intT<type16b *>(target);	
}

void HI_checksum_set_int32b(type32b * target) {
	HI_checksum_set_intT<type32b *>(target);	
}

void HI_checksum_set_int64b(type64b * target) {
	HI_checksum_set_intT<type64b *>(target);	
}

void HI_checksum_set_float(float * target) {
	HI_checksum_set_floatT<float *>(target);	
}

void HI_checksum_set_double(double * target) {
	HI_checksum_set_floatT<double *>(target);	
}


int HI_checksum_check_int8b(type8b * target) {
	int error = 0;
	error = HI_checksum_check_intT<type8b *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on int8_t data!\n");
	}
#endif
	return error;
}

int HI_checksum_check_int16b(type16b * target) {
	int error = 0;
	error = HI_checksum_check_intT<type16b *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on int16_t data!\n");
	}
#endif
	return error;
}

int HI_checksum_check_int32b(type32b * target) {
	int error = 0;
	error = HI_checksum_check_intT<type32b *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on int32_t data!\n");
	}
#endif
	return error;
}

int HI_checksum_check_int64b(type64b * target) {
	int error = 0;
	error = HI_checksum_check_intT<type64b *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on int64_t data!\n");
	}
#endif
	return error;
}

int HI_checksum_check_float(float * target) {
	int error = 0;
	error = HI_checksum_check_floatT<float *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on float data!\n");
	}
#endif
	return error;
}

int HI_checksum_check_double(double * target) {
	int error = 0;
	error = HI_checksum_check_floatT<double *>(target);	
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		fprintf(stderr, "====> Checksum Error detected on double data!\n");
	}
#endif
	return error;
}

void HI_checkpoint_register(const void * target, long int size, int option) {
	cp_sizemap[target] = size;
	cp_optionmap[target] = option;
	if( option == 0 ) {
		void *cp_data = malloc(size);
		cp_checkpointmap[target] = cp_data; //map target to checkpoint data
		//fprintf(stderr, "HI_checkpoint_register() is called\n");
	}
}

void HI_checkpoint_backup(const void * target) {
	long int size;
	int option;
	void *cp_data;
	long int i;
	if( (cp_sizemap.count(target) == 0) || (cp_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checkpoint_backup()]\n");
		exit(1);
	} else {
		//fprintf(stderr, "HI_checkpoint_backup() is called\n");
		size = cp_sizemap[target];
		option = cp_optionmap[target];
		cp_data = cp_checkpointmap[target];	
		if( option == 0 ) {
			memcpy(cp_data, target, size);
		}
	}
}

void HI_checkpoint_restore(void * target) {
	long int size;
	int option;
	void *cp_data;
	long int i;
	if( (cp_sizemap.count(target) == 0) || (cp_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checkpoint_restore()]\n");
		exit(1);
	} else {
		//fprintf(stderr, "HI_checkpoint_restore() is called\n");
		size = cp_sizemap[target];
		option = cp_optionmap[target];
		cp_data = cp_checkpointmap[target];	
		if( option == 0 ) {
			memcpy(target, (const void *)cp_data, size);
		}
	}
}

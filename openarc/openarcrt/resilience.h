#ifndef __RESILIENCE_HEADER__

#define __RESILIENCE_HEADER__

///////////////////////////////////////
//Functions used for resilience test //
///////////////////////////////////////
#define numFloatBits	32.0
#define numDoubleBits	64.0
////////////////////////////////////////////////////////////////////
//User may need to modify below typedef so that type32b refers to //
//unsigned int type of 4 bytes and type64b refers to unsigned int //
//type of 8 bytes.                                                //
////////////////////////////////////////////////////////////////////
//Example: on Newark (x86_64 GNU/Linux) machine: 
//	sizeof(unsigned int) = 4 
//	sizeof(unsigned long int) = 8
//	sizeof(unsigned long long int) = 8
typedef unsigned char type8b;
typedef unsigned short int type16b;
typedef unsigned int type32b;
typedef unsigned long int type64b;
typedef signed char type8bS;
typedef signed short int type16bS;
typedef signed int type32bS;
typedef signed long int type64bS;

typedef union {
	float f;
	type32b i;
} FloatBits;

typedef union {
	double d;
	type64b i;
} DoubleBits;


// Set a new random seed value for the internal random-number generator.
// This function is internally called by acc_init() function, and thus 
// we don't need to call this explicitly if the target program is a CUDA 
// program translated from OpenACC.
extern void HI_set_srand();
// Generate an 8-bit vector that contains random 1s specified 
// by the input (numFaults).
extern type8b HI_genbitvector8b(int numFaults);
// Generate an 16-bit vector that contains random 1s specified 
// by the input (numFaults).
extern type16b HI_genbitvector16b(int numFaults);
// Generate an 32-bit vector that contains random 1s specified 
// by the input (numFaults).
extern type32b HI_genbitvector32b(int numFaults);
// Generate an 64-bit vector that contains random 1s specified 
// by the input (numFaults).
extern type64b HI_genbitvector64b(int numFaults);
// Generate a random number between [0, Range-1]
extern unsigned long int HI_genrandom_int(unsigned long int Range);
// Sort values in an input array (iArray) of size iSize in an increasing order.
extern void HI_sort_int(unsigned int *iArray, int iSize);
///////////////////////////////////////////////////////////////////////
// Flip bits in the input data (target) by XORing target with bitvec //
// Type of target should have the same bits as bitvec.               //
///////////////////////////////////////////////////////////////////////
// target: input data where faults will be injected.
// ftinject: performs actual fault injection only if this value is non-zero.
// epos: contains an index for the input array data (target) to decide which 
//       element of the array should be fault-injected.
//       If target is scalar, this should be 0.
extern void HI_ftinjection_int8b(type8b * target, int ftinject,  long int epos, type8b bitvec);
extern void HI_ftinjection_int16b(type16b * target, int ftinject,  long int epos, type16b bitvec);
extern void HI_ftinjection_int32b(type32b * target, int ftinject,  long int epos, type32b bitvec);
extern void HI_ftinjection_int64b(type64b * target, int ftinject,  long int epos, type64b bitvec);
extern void HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec);
extern void HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec);

extern void HI_checksum_register_intT(const void * target, long int size, int option);
extern void HI_checksum_register_floatT(const void * target, long int size, int option);

extern void HI_checksum_set_int8b(type8b * target);
extern void HI_checksum_set_int16b(type16b * target);
extern void HI_checksum_set_int32b(type32b * target);
extern void HI_checksum_set_int64b(type64b * target);
extern void HI_checksum_set_float(float * target);
extern void HI_checksum_set_double(double * target);

extern int HI_checksum_check_int8b(type8b * target);
extern int HI_checksum_check_int16b(type16b * target);
extern int HI_checksum_check_int32b(type32b * target);
extern int HI_checksum_check_int64b(type64b * target);
extern int HI_checksum_check_float(float * target);
extern int HI_checksum_check_double(double * target);

extern void HI_checkpoint_register(const void * target, long int size, int option);
extern void HI_checkpoint_backup(const void * target);
extern void HI_checkpoint_restore(void * target);

#endif

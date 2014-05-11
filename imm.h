#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
using std::vector;
#include <string>
using std::string;

//Parameter Definitions
#define IMM_MAX_ORDER 8

//Testing functions
__host__ __device__ void build_distribution(int * model, char * sequence, int length, int *output);
__host__ __device__ void build_chi2_table(int * dist1, int * dist2, int * output, int length);
__host__ __device__ float chi_squared_score(int * table, int length);
__device__ __host__ float score_order_pair(int * model, char * sequence, int order);

/*
This class represents an Interpolated Markov Model
stored as a list of counts for each position in the window
[A,C,G,T,AA...TT,AAA...TTT,etc.]
*/
class IMM {
protected:
	int order, window;
	int *d_counts;
	size_t order_sum;
	size_t total_bytes;
	
public:
	IMM();
	~IMM();

	void init(int window, int order);
	void add(vector<string> & sequences);
	void score(vector<string> & sequences);
	void dump(vector<int> & result);
	void dispose();
};
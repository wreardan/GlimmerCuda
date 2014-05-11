#include "imm.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <algorithm>
#include <sstream>

using namespace std;

static string bases = "ACGT";
string bases_to_indices(string sequence) {
	string result;
	for(char c : sequence) {
		int index = bases.find(c);
		result.push_back(index);
	}
	return result;
}


IMM::IMM() {
	order = -1;
	window = -1;
	d_counts = NULL;
}

void IMM::dispose() {
	if(d_counts != NULL) {
		cudaFree(d_counts);
		d_counts = NULL;
		order = -1;
	}
}


IMM::~IMM() {
	dispose();
}

/* Function to calculate x raised to the power y */
//http://www.geeksforgeeks.org/write-a-c-program-to-calculate-powxn/
int power(int x, unsigned int y)
{
    if( y == 0)
        return 1;
    else if (y%2 == 0)
        return power(x, y/2)*power(x, y/2);
    else
        return x*power(x, y/2)*power(x, y/2);
 
}

//window is the size of the sliding window
//order is the markov-order of the model
void IMM::init(int window, int order) {
	this->order = order;
	this->window = window;

	order_sum = 0;
	for(int i = 0; i <= order; i++) {
		order_sum += power(4, i+1);
	}

	total_bytes = window * order_sum * sizeof(int);
	printf("%d, %d\n", order_sum, total_bytes);
	
    cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&d_counts, total_bytes);
	assert (cudaSuccess == cudaStatus);
	cudaStatus = cudaMemset(d_counts, 0, total_bytes);
	assert (cudaSuccess == cudaStatus);
}

__global__ void counting_kernel(int *model, char * sequences, int max_order, int window) {
    int num = threadIdx.x; //sequence number
	int order = threadIdx.y; //order number
    
	//get sequence
	char * sequence = sequences + num * window;

	//get index of sequence
	int index = 0;
	for(int i = 0; i < window; i++) {
		index = index * 4 + *(sequence+i);
	}

	//add to model
	//int model_index = order * window * sizeof(int);
	//TODO: Multiple orders
	model += index;
	atomicAdd(model, 1);
}

//Add Sequences to the Model
void IMM::add(vector<string> sequences) {
    cudaError_t cudaStatus;
	//Concatenate sequences
    stringstream ss;
	for(string & seq : sequences) {
		ss << seq.substr(0, window);
	}

    //for_each(sequences.begin(), sequences.end(), [&ss] (const string& s) { ss << s; });
	string all = bases_to_indices(ss.str());
	int size = all.size();
	printf("total_length=%d\n", size);
	
	//Send sequences to GPU
	char *d_seq;
	cudaStatus = cudaMalloc(&d_seq, size);
	assert (cudaSuccess == cudaStatus);
	cudaMemcpy(d_seq, &all[0], size, cudaMemcpyHostToDevice);
	assert (cudaSuccess == cudaStatus);

	//invoke counting kernel
    counting_kernel<<<1, size>>>(d_counts, d_seq, order, window);

	//Cleanup
	cudaFree(d_seq);
}

void IMM::dump(vector<int> & result) {
	//assert(d_counts
	result.clear();
}

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

//http://www.bedroomlan.org/writing/oric-c-programming/iteration-vs-recursion
__host__ __device__ unsigned int power (unsigned int x, unsigned int y)
{
	unsigned int result;
        if (y == 0) return 1;
	for (result = x; y > 1; y--) result *= x;
	return result;
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

__global__ void counting_kernel(int *model, char * sequences, int pos_size, int max_order, int window) {
    int num = threadIdx.x; //sequence number
	int position = threadIdx.y; //position index
	int order = threadIdx.z; //order number

	if(position + order >= window) {
		return;
	}
    
	//get sequence
	char * sequence = sequences + num * window + position;

	//compute index, order_index
	int index = 0;
	for(int i = 0; i < order+1; i++) {
		index = index * 4 + *(sequence+i);
	}
	int order_index = 0;
	for(int i = 0; i < order; i++) {
		order_index += power(4, i+1);
	}

	//increment count
	int * count = model + index + order_index + pos_size * order;
	count += position * pos_size;
	atomicAdd(count, 1);
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
	int num_sequences = sequences.size();
	dim3 threads_per_block(num_sequences,2,order+1);
	dim3 blocks(1,1,1);
    counting_kernel<<<blocks, threads_per_block>>>(d_counts, d_seq, order_sum, order, window);

	//Cleanup
	cudaFree(d_seq);
}

void IMM::dump(vector<int> & result) {
	//setup result vector
	result.clear();
	int arr_size = total_bytes / sizeof(int);
	result.resize(arr_size);

	//copy data from gpu
	cudaMemcpy(&result[0], d_counts, total_bytes, cudaMemcpyDeviceToHost);
}

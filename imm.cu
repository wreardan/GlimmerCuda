#include "imm.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <algorithm>
#include <sstream>

using namespace std;


static string bases = "ACGT";
//convert bases to indices
//i.e. "ACGTGCA" -> 0123210
string bases_to_indices(string sequence) {
	string result;
	for(char c : sequence) {
		int index = bases.find(c);
		//TODO: something more reasonable here?
		if(index == string::npos) {
			index = rand() % bases.size();
		}

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


//Take x to power y
//http://www.bedroomlan.org/writing/oric-c-programming/iteration-vs-recursion
__host__ __device__ unsigned int power (unsigned int x, unsigned int y)
{
	unsigned int result;

	if (y == 0) {
		return 1;
	}

	for (result = x; y > 1; y--) {
		result *= x;
	}

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
	
    cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&d_counts, total_bytes);
	assert (cudaSuccess == cudaStatus);
	cudaStatus = cudaMemset(d_counts, 0, total_bytes);
	assert (cudaSuccess == cudaStatus);
}

__host__ __device__ int get_sequence_index(char * sequence, int length) {
	int index = 0;
	for(int i = 0; i < length; i++) {
		index = index * 4 + *(sequence+i);
	}
	return index;
}



//this kernel builds the imm from a set of training sequences
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
	int index = get_sequence_index(sequence, order+1);
	int order_index = 0;
	for(int i = 0; i < order; i++) {
		order_index += power(4, i+1);
	}

	//increment count
	int * count = model + index + order_index + pos_size * order;
	count += position * pos_size;
	atomicAdd(count, 1);
}


void send_windows_to_gpu(vector<string> & sequences, int window, char **d_seq) {
    cudaError_t cudaStatus;
	//Concatenate sequences
    stringstream ss;
	for(string & seq : sequences) {
		ss << seq.substr(0, window);
	}

	string all = bases_to_indices(ss.str());
	int size = all.size();
	
	//Send sequences to GPU
	cudaStatus = cudaMalloc(d_seq, size);
	assert (cudaSuccess == cudaStatus);
	cudaMemcpy(*d_seq, &all[0], size, cudaMemcpyHostToDevice);
	assert (cudaSuccess == cudaStatus);
}



//Add Sequences to the Model
void IMM::add(vector<string> & sequences) {
	char *d_seq;
	send_windows_to_gpu(sequences, window, &d_seq);
	
	//invoke counting kernel
	int num_sequences = sequences.size();
	dim3 threads_per_block(num_sequences,window,order+1);
	dim3 blocks(1,1,1);
    counting_kernel<<<blocks, threads_per_block>>>(d_counts, d_seq, order_sum, order, window);

	//Cleanup
	cudaFree(d_seq);
}

//The MEAT of the matter: Scoring


//Compute the chi^2 score of a chi2 table
__device__ __host__ float chi_squared_score(int * table, int length) {
	//sum counts in table
	float N = 0.0f;
	for(int i = 0; i < length*2; i++) {
		N += table[i];
	}
	if (N == 0.0f) {
		return 0.0f;
	}
	//Compute Score
	float score = 0.0f;
	for(int i = 0; i < length; i++) {
		float Ri = table[i*2+0] + table[i*2+1];
		for(int j = 0; j < 2; j++) {
			float Cj = 0.0f;
			for(int x = 0; x < length; x++) {
				Cj += table[x*2+j];
			}
			float Eij = Ri*Cj/N;
			float Oij = table[i*2+j];
			if(Eij != 0.0f) {
				score += (Oij - Eij)*(Oij - Eij)/Eij;
			}
		}
	}
	return score;
}


//build two distributions into a chi^2 table
__device__ __host__ void build_chi2_table(int * dist1, int * dist2, int * output, int length) {
	for(int i = 0; i < length; i++) {
		output[i*2] = dist1[i];
		output[i*2+1] = dist2[i];
	}
}

//Test to see if the chi_squared_score and build_chi2_table methods work correctly
bool test_chi_squared_test() {
	int dist1[] = {25,9,11,17};
	int dist2[] = {1,1,59,1};
	int table[8];
	int expected_table[] = {25,1,9,1,11,59,17,1};

	build_chi2_table(dist1, dist2, table, 4);
	float result = chi_squared_score(table, 4);
	int eq = memcmp(table, expected_table, sizeof(table));
	assert(eq == 0);
	return (abs(result - 75.69f) < 0.001);
}


//Build a distribution from an order of the model based on a subsequence
__device__ __host__ void build_distribution(int * model, char * sequence, int length) {
	
}



__global__ void scoring_kernel(int *model, char * sequences, float * scores) {
    int num = threadIdx.x; //sequence number
	int position = threadIdx.y; //position index
	int order = threadIdx.z; //order number

	//Compute lambdas at position based on model

	//Score character based on lambdas
}


void IMM::score(vector<string> & sequences) {
	//send sequences to gpu
	char *d_seq;
	send_windows_to_gpu(sequences, window, &d_seq);

	//Score Positions
	int num_sequences = sequences.size();
	dim3 threads_per_block(num_sequences,2,order+1);
	dim3 blocks(1,1,1);
    counting_kernel<<<blocks, threads_per_block>>>(d_counts, d_seq, order_sum, order, window);

	//Cleanup
	cudaFree(d_seq);
}


//Dump model to a vector of ints
void IMM::dump(vector<int> & result) {
	//setup result vector
	result.clear();
	int arr_size = total_bytes / sizeof(int);
	result.resize(arr_size);

	//copy data from gpu
	cudaMemcpy(&result[0], d_counts, total_bytes, cudaMemcpyDeviceToHost);
}

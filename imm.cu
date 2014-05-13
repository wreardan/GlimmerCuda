#include "imm.h"

#include <assert.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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
	d_chi2_pvalue_table = NULL;
	order_sum = 0;
	total_bytes = 0;
}


void IMM::dispose() {
	if(d_counts != NULL) {
		cudaFree(d_counts);
		d_counts = NULL;
		order = -1;
	}
	if(d_chi2_pvalue_table != NULL) {
		cudaFree(d_chi2_pvalue_table);
		d_chi2_pvalue_table = NULL;
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


void send_windows_to_gpu(vector<string> & sequences, int window, char **d_seq) {
    cudaError_t cuda_status;
	//Concatenate sequences
    stringstream ss;
	for(string & seq : sequences) {
		assert(seq.size() >= window);
		ss << seq.substr(0, window);
	}

	string all = bases_to_indices(ss.str());
	int size = all.size();
	
	//Send sequences to GPU
	cuda_status = cudaMalloc(d_seq, size);
	assert (cudaSuccess == cuda_status);
	cudaMemcpy(*d_seq, &all[0], size, cudaMemcpyHostToDevice);
	assert (cudaSuccess == cuda_status);
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
	
    cudaError_t cuda_status;
	cuda_status = cudaMalloc(&d_counts, total_bytes);
	assert (cudaSuccess == cuda_status);
	cuda_status = cudaMemset(d_counts, 0, total_bytes);
	assert (cudaSuccess == cuda_status);
}

__host__ __device__ int get_sequence_index(char * sequence, int length) {
	int index = 0;
	for(int i = 0; i < length; i++) {
		index = index * 4 + *(sequence+i);
	}
	return index;
}


__host__ __device__ int get_order_index(int order) {
	int order_index = 0;
	for(int i = 0; i < order; i++) {
		order_index += power(4, i+1);
	}
	return order_index;
}


__host__ __device__ int get_sequence_order_index(char * sequence, int order) {
	int index = get_sequence_index(sequence, order + 1);
	int order_index = get_order_index(order);
	return index + order_index;
}


__host__ __device__ int get_dist_sequence_order_index(char * sequence, int order) {
	int index = get_sequence_index(sequence, order) * 4;
	int order_index = get_order_index(order);
	return index + order_index;
}


//this kernel builds the imm from a set of training sequences
__global__ void counting_kernel(int *model, char * sequences, int pos_size, int max_order, int window) {
    int num = threadIdx.x; //sequence number
	int position = blockIdx.y; //position index
	int order = blockIdx.z; //order number

	if(position + order >= window) {
		return;
	}
    
	//get sequence
	char * sequence = sequences + num * window + position;

	//compute index, order_index
	int index = get_sequence_order_index(sequence, order);

	//increment count
	int * count = model + index;
	count += (position+order) * pos_size;
	atomicAdd(count, 1);
}


//Add Sequences to the Model
void IMM::add(vector<string> & sequences) {
	char *d_seq;
	send_windows_to_gpu(sequences, window, &d_seq);

	//wait for device
	cudaDeviceSynchronize();
	
	//invoke counting kernel
	int num_sequences = sequences.size();
	dim3 threads_per_block(num_sequences,1,1);
	dim3 blocks(1,window,order+1); //TODO: +1 needed??
    counting_kernel<<<blocks, threads_per_block>>>(d_counts, d_seq, order_sum, order, window);
	cudaError cuda_status = cudaGetLastError();
	if ( cudaSuccess != cuda_status ) {
		printf("IMM:add counting_kernel failed to execute with error: %d\n", cuda_status);
	}

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

//Build a distribution from an order of the model based on a subsequence
__device__ __host__ void build_distribution(int * model, char * sequence, int order, int *output) {
	//TODO: take position into account, or input custom model pointer
	int index = get_dist_sequence_order_index(sequence, order);

	for(int b = 0; b < 4; b++) {
		output[b] = model[index+b];
	}
}


//Performs a Chi^2 test on a pair of orders (order, order+1) with sequence at index
__device__ __host__ float score_order_pair(int * model, char * sequence, int order, int * next_order_count) {
	//create distribution with lower-order model
	int lower_order_dist[4];
	build_distribution(model, sequence, order, lower_order_dist);
	//create distribution with order+1 model
	int higher_order_dist[4];
	build_distribution(model, sequence-1, order+1, higher_order_dist);
	//chi-squared test
	int table[8];
	build_chi2_table(lower_order_dist, higher_order_dist, table, 4);
	float score = chi_squared_score(table, 4);
	//calculate next-order count
	*next_order_count = 0;
	for(int i = 0; i < 4; i++) {
		*next_order_count += higher_order_dist[i];
	}

	return score;
}


//Computes lambdas then score the sequence
__global__ void scoring_kernel(int *model, char * sequences, float * scores, int window, int max_order, int pos_size, float * pvalues) {
	char * sequence_position;

    int num = threadIdx.x + blockIdx.x * blockDim.x; //sequence number
	int position = blockIdx.y; //position index
    
	//get sequence
	char * sequence = sequences + num * window;

	//Compute lambdas at position based on model
	float lambdas[IMM_MAX_ORDER] = {1.0};
	model += position * pos_size;
	
	max_order = min(max_order, position);
	for(int order = 0; order < max_order; order++) {
		int next_order_count;
		sequence_position = sequence + position - order;
		float chi2_score = score_order_pair(model, sequence_position, order, &next_order_count);
		int chi2_index = (int)(chi2_score * 10.0f);
		chi2_index = min(chi2_index, 50);
		float probability = pvalues[chi2_index];
		float d = 1.0f - probability;
		float value;
		if(next_order_count > IMM_MIN_COUNT) {
			value = 1.0f;
		} else if(d > 0.5f) {
			value = d * next_order_count / 40.0f;
		} else {
			value = 0.0f;
		}
		lambdas[order+1] = value;
		//printf("%d, %d, %d, %f\n", num, order+1, position, value);
		//printf("num: %d, order: %d, position: %d, lambda: %f\n", num, order+1, position, value);
	}

	//Score character based on lambdas
	float score = 0.0f;
	float weight = 1.0f;
	for(int order = max_order; order >= 0; order--) {
		sequence_position = sequence + position - order;
		int index = get_sequence_order_index(sequence_position, order);
		int count = model[index];

		int dist_index = get_dist_sequence_order_index(sequence_position, order);
		int sum = 0;
		for(int base = 0; base < 4; base++) {
			int c = model[dist_index + base];
			sum += c;
		}
		//printf("%d %d %d %d\n", position, order, count, sum);
		float probability = (count + 1.0f) / (sum + 4.0f);
		float weighted = lambdas[order] * weight;
		//printf("weighted: %f\n", weighted);
		score += probability * weighted;
		weight *= 1 - lambdas[order];
	}
	//printf("num: %d, position: %d, score: %f\n", num, position, log10(score));

	int score_index = num*window + position;
	scores[score_index] = log10(score);
}


void IMM::score(vector<string> & sequences, vector<float> & scores) {
    cudaError_t cuda_status;
	//assertions
	assert(d_chi2_pvalue_table != NULL);
	assert(order <= IMM_MAX_ORDER);
	assert(window >= order);

	//allocate memory for scores
	float *d_scores;
	size_t size = sequences.size() * window * sizeof(float);
	cuda_status = cudaMalloc(&d_scores, size);
	assert (cudaSuccess == cuda_status);

	//send sequences to gpu
	char *d_seq;
	send_windows_to_gpu(sequences, window, &d_seq);

	//Score Positions
	int num_sequences = sequences.size();
	dim3 threads_per_block(num_sequences,1,1);
	dim3 blocks(1,window,1);
    scoring_kernel<<<blocks, threads_per_block>>>(d_counts, d_seq, d_scores, window, order, order_sum, d_chi2_pvalue_table);
	
	//wait for device
	cudaDeviceSynchronize();

	//sum scores, use Thrust library?
	float * raw_scores = new float[size];
	cuda_status = cudaMemcpy(raw_scores, d_scores, size, cudaMemcpyDeviceToHost);
	assert (cudaSuccess == cuda_status);

	for(int i = 0; i < num_sequences; i++) {
		float score = 0.0f;
		for(int pos = 0; pos < window; pos++) {
			score += raw_scores[i * window + pos];
			//printf("position: %d, score: %f\n", pos, score);
		}
		//printf("sequence: %d, score: %f\n", i, score);
		scores.push_back(score);
	}


	//Cleanup
	delete raw_scores;
	cudaFree(d_seq);
	cudaFree(d_scores);
}


//Load chi2->Probability values from file
void IMM::load_pvalues(string & filename) {
    cudaError_t cuda_status;
	vector<float> pvalues;
	ifstream file(filename);
	string line;

	getline(file, line); //skip first line
	while(file.good()) {
		float chi2, pvalue;
		file >> chi2;
		file >> pvalue;
		pvalues.push_back(pvalue);
	}

	//allocate memory then send to gpu
	size_t size = pvalues.size() * sizeof(float);
	cuda_status = cudaMalloc(&d_chi2_pvalue_table, size);
	assert (cudaSuccess == cuda_status);
	cuda_status = cudaMemcpy(d_chi2_pvalue_table, &pvalues[0], size, cudaMemcpyHostToDevice);
	assert (cudaSuccess == cuda_status);

	//cleanup
	file.close();
}


void read_fasta(vector<string> & sequences, string & filename) {
	//vector<string> descriptions;
	ifstream file(filename);
	stringstream ss;
	string sequence;
	string description;

	while(file.good()) {
		string line;
		getline(file, line);
		if(line.size() <= 0) {
			continue;
		} else if(line[0] == '>') {
			if(description.size() > 0) {
				sequences.push_back(sequence);
				//sequences.push_back(description);
			}
			description = line.substr(1);
			sequence.clear();
		} else {
			sequence.append(line);
		}
	}
	if(sequence.size() > 0) {
		sequences.push_back(sequence);
	}
	//cleanup
	file.close();
}


void read_sequences(vector<string> & sequences, string & filename) {
	ifstream file(filename);
	while(file.good()) {
		string sequence;
		getline(file, sequence);
		if(sequence.size() > 0) {
			sequences.push_back(sequence);
		}
	}
	file.close();
}



//Dump model to a vector of ints
void IMM::dump(vector<int> & result) {
    cudaError_t cuda_status;
	//wait for device
	cudaDeviceSynchronize();

	//setup result vector
	result.clear();
	int arr_size = total_bytes / sizeof(int);
	result.resize(arr_size);

	//copy data from gpu
	cuda_status = cudaMemcpy(&result[0], d_counts, total_bytes, cudaMemcpyDeviceToHost);
	assert (cudaSuccess == cuda_status);
}

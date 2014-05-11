#include "sequence.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

Sequence::Sequence() {
	d_sequence = NULL;
	length = 0;
}

Sequence::~Sequence() {
	dispose();
}

void Sequence::load(string sequence) {
    cudaError_t cudaStatus;
	length = sequence.length();
    cudaStatus = cudaMalloc((void**)&d_sequence, length );
	assert (cudaStatus == cudaSuccess);
	cudaStatus = cudaMemcpy(d_sequence, &sequence[0], length, cudaMemcpyHostToDevice);
	assert (cudaStatus == cudaSuccess);
}

void Sequence::dispose() {
	if(d_sequence != NULL) {
		cudaFree(d_sequence);
		d_sequence = NULL;
		length = 0;
	}
}

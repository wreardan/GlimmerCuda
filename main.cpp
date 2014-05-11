
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include "sequence.h"
#include "imm.h"

int main()
{
	Sequence sequence;
	sequence.load("ACGTTACGA");

	IMM imm;
	imm.init(1, 0);
	
	vector<string> sequences;
	sequences.push_back("ATGTATACGGGATAA");
	sequences.push_back("TTGGGGATAGAGGAA");
	imm.add(sequences);

    return 0;
}
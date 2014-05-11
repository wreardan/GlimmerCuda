
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>


#include "sequence.h"

#include "imm_test.h"

int main()
{
	Sequence sequence;
	sequence.load("ACGTTACGA");

	//assert(test_imm_1());
	assert(test_imm_2());

    return 0;
}
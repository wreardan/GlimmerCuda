#include <assert.h>

#include "imm_test.h"
#include "imm.h"


void imm_test() {
	assert(test_imm_1());
	assert(test_imm_2());

	assert(test_chi_squared_test());

	printf("imm_test() completed succesfully, congrats!\n");
}


bool test_imm_1() {
	IMM imm;
	imm.init(1, 0);
	
	vector<string> sequences;
	sequences.push_back("A");
	sequences.push_back("A");
	sequences.push_back("A");
	sequences.push_back("C");
	sequences.push_back("C");
	sequences.push_back("T");
	imm.add(sequences);

	vector<int> actual;
	imm.dump(actual);

	int arr[] = {3,2,0,1};
	vector<int> expected(arr, arr + sizeof(arr) / sizeof(arr[0]));

	return expected == actual;
}


bool test_imm_2() {
	IMM imm;
	imm.init(2, 1);

	vector<string> sequences;
	sequences.push_back("AA");
	sequences.push_back("AA");
	sequences.push_back("AC");
	sequences.push_back("AG");
	sequences.push_back("AT");
	sequences.push_back("CC");
	sequences.push_back("GG");
	sequences.push_back("TT");

	imm.add(sequences);

	vector<int> actual;
	imm.dump(actual);

	int arr[] = {
		5,1,1,1,  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		2,2,2,2,  2,1,1,1, 0,1,0,0, 0,0,1,0, 0,0,0,1,
	};
	vector<int> expected(arr, arr + sizeof(arr) / sizeof(arr[0]));

	return expected == actual;
}

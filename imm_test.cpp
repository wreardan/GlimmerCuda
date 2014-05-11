#include <assert.h>

#include "imm_test.h"
#include "imm.h"

#define assert_memory_equal(a,b) assert(memcmp(a,b,sizeof(a)) == 0)


void imm_test() {
	assert(test_imm_1());
	assert(test_imm_2());

	assert(test_chi_squared_test());
	assert(test_build_distribution());

	printf("imm_test() completed succesfully, congrats!\n");
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


bool test_build_distribution() {
	int model[] = {
		//Distribution from second position
		2,2,2,2,  2,1,1,1, 0,1,0,0, 0,0,1,0, 0,0,0,1,
	};
	int dist[4]= {0,0,0,0};
	
	char * sequence = "\1";
	build_distribution(model, sequence, 2, dist);
	int expected[4] = {0,1,0,0};
	assert_memory_equal(dist, expected);

	sequence = "\3";
	build_distribution(model, sequence, 2, dist);
	int expected2[] = {0,0,0,1};
	assert_memory_equal(dist, expected2);
	
	build_distribution(model, sequence, 1, dist);
	int expected0[] = {2,2,2,2};
	assert_memory_equal(dist, expected0);

	sequence = "\0";
	float chi2 = score_order_pair(model, sequence, 0);
	assert(abs(chi2 - 0.325f) < 0.001f);

	return true;
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

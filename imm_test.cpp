#include <assert.h>

#include "imm_test.h"
#include "imm.h"

#define assert_memory_equal(a,b) assert(memcmp(a,b,sizeof(a)) == 0)


void imm_test() {
	//assert(test_imm_1());
	//assert(test_imm_2());
	//assert(test_imm_3());
	assert(test_imm_4());

	//assert(test_chi_squared_test());
	//assert(test_build_distribution());

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
	int next_order_count;
	float chi2 = score_order_pair(model, sequence, 0, &next_order_count);
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

void test_setup_2(IMM & imm) {
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

	string filename = "chisquare_df3_pvalues";
	imm.load_pvalues(filename);
}



bool test_imm_2() {
	IMM imm;
	test_setup_2(imm);

	vector<int> actual;
	imm.dump(actual);

	int arr[] = {
		//First Position
		5,1,1,1,  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		//Second Position
		2,2,2,2,  2,1,1,1, 0,1,0,0, 0,0,1,0, 0,0,0,1,
	};
	vector<int> expected(arr, arr + sizeof(arr) / sizeof(arr[0]));

	return expected == actual;
}


//a more interesting test using actual data
bool test_imm_3() {
	IMM imm;
	string filename;

	imm.init(9, 5);
	filename = "chisquare_df3_pvalues";
	imm.load_pvalues(filename);

	vector<string> training_sequences;
	filename = "iterated.train";
	read_fasta(training_sequences, filename);
	//training_sequences.resize(500);
	imm.add(training_sequences);

	vector<int> dumped;
	imm.dump(dumped);
	
	vector<string> test_sequences;
	test_sequences.push_back(string("ATGATTTGA"));
	//test_sequences.push_back(string("GCGCGCCGCGCG"));
	//imm.score(test_sequences);

	return true;
}

bool test_imm_4() {
	IMM imm;
	string filename;

	imm.init(9, 5);
	filename = "chisquare_df3_pvalues";
	imm.load_pvalues(filename);

	vector<string> training_sequences;
	filename = "hw3_train_real";
	read_sequences(training_sequences, filename);
	imm.add(training_sequences);

	vector<int> dumped;
	imm.dump(dumped);

	/*
	for(int i = 0; i < dumped.size(); i++) {
		printf("%d\n", dumped[i]);
	}
	*/
	
	vector<string> test_sequences;
	vector<float> scores;
	filename = "hw3_test_real";
	read_sequences(test_sequences, filename);
	test_sequences.resize(1);
	imm.score(test_sequences, scores);
	
	/*
	for(int i = 0; i < scores.size(); i++) {
		printf("%f\n", scores[i]);
	}
	*/

	return true;
}

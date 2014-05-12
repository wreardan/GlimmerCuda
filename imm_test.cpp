#include <assert.h>

#include "imm_test.h"
#include "imm.h"

#define assert_memory_equal(a,b) assert(memcmp(a,b,sizeof(a)) == 0)


void imm_test() {
	/*assert(test_imm_1());
	assert(test_imm_2());
	assert(test_imm_3());*/
	//test_imm_start_codons();
	assert(test_imm_4());

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
	build_distribution(model, sequence, 1, dist);
	int expected[4] = {0,1,0,0};
	assert_memory_equal(dist, expected);

	sequence = "\3";
	build_distribution(model, sequence, 1, dist);
	int expected2[] = {0,0,0,1};
	assert_memory_equal(dist, expected2);
	
	build_distribution(model, sequence, 0, dist);
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

bool test_imm_start_codons() {
	IMM imm;
	imm.init(3, 2);

	vector<string> sequences;
	sequences.push_back("ATG");
	sequences.push_back("GTG");
	sequences.push_back("TTG");
	sequences.push_back("ATT");
	sequences.push_back("CTG");

	imm.add(sequences);

	string filename = "chisquare_df3_pvalues";
	imm.load_pvalues(filename);

	vector<int> actual;
	imm.dump(actual);

	int arr[] = {
		//First Position
		2,1,1,1, //0th order
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, //1st order

		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, //2nd order
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		//Second Position
		0,0,0,5, //0th order
		
		0,0,0,2, 0,0,0,1, 0,0,0,1, 0,0,0,1, //1st order

		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, //2nd order
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
		//Third Position
		0,0,4,1, //0th order
		
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,4,1, //1st order

		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,1, //2nd order
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0,
		0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0,
	};
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
	IMM positive_imm, negative_imm;
	string filename;

	positive_imm.init(9, 5);

	vector<string> train_real;
	filename = "hw3_train_real";
	read_sequences(train_real, filename);
	//training_sequences.resize(1);
	positive_imm.add(train_real);

	vector<int> dumped;
	positive_imm.dump(dumped);
	
	negative_imm.init(9, 5);
	vector<string> train_false;
	filename = "hw3_train_false";
	read_sequences(train_false, filename);
	//training_sequences.resize(1);
	negative_imm.add(train_false);
	
	/*
	for(int i = 0; i < 24; i++) {
		printf("%d %d\n", i, dumped[i]);
	}
	for(int i = 5460; i < 5460+24; i++) {
		printf("%d %d\n", i, dumped[i]);
	}
	for(int i = 5460*2; i < 5460*2+24; i++) {
		printf("%d %d\n", i, dumped[i]);
	}
	*/
	filename = "chisquare_df3_pvalues";
	positive_imm.load_pvalues(filename);
	negative_imm.load_pvalues(filename);
	
	vector<string> test_sequences;
	vector<float> scores;
	filename = "hw3_test_real";
	read_sequences(test_sequences, filename);
	test_sequences.resize(10);

	positive_imm.score(test_sequences, scores);
	
	vector<float> negative_scores;
	negative_imm.score(test_sequences, negative_scores);
	
	for(int i = 0; i < scores.size(); i++) {
		printf("%f\n", scores[i] - negative_scores[i]);
	}
	

	return true;
}

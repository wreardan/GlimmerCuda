#pragma once

#include <vector>
using std::vector;
#include <string>
using std::string;

//Testing functions
bool test_chi_squared_test();

/*
This class represents an Interpolated Markov Model
stored as a list of counts for each position in the window
[A,C,G,T,AA...TT,AAA...TTT,etc.]
*/
class IMM {
protected:
	int order, window;
	int *d_counts;
	size_t order_sum;
	size_t total_bytes;
	
public:
	IMM();
	~IMM();

	void init(int window, int order);
	void add(vector<string> & sequences);
	void score(vector<string> & sequences);
	void dump(vector<int> & result);
	void dispose();
};
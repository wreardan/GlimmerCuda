#pragma once

#include <vector>
using std::vector;
#include <string>
using std::string;

/*
This class represents an Interpolated Markov Model
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
	void add(vector<string> sequences);
	void dispose();
	void dump(vector<int> & result);
};
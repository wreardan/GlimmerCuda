#pragma once

#include <vector>
using std::vector;
#include <string>
using std::string;

class Sequence {

protected:
	char * d_sequence;
	size_t length;

public:
	Sequence();
	~Sequence();

	void load(string sequence);
	void dispose();
};
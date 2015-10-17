#include <iostream>
#include "Start.h"
#include "StateSpace.h"
#include "Data.h"

using namespace std;
using namespace DNest3;

int main(int argc, char** argv)
{
	// Load the data
	Data::get_instance().load("data.txt");

	MTSampler<StateSpace> sampler = setup_mt<StateSpace>(argc, argv);
	sampler.run();

	return 0;
}


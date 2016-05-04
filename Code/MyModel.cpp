#include "MyModel.h"
#include "DNest4/code/DNest4.h"

using namespace std;
using namespace DNest4;

MyModel::MyModel()
:modes(2, 100, false,
        MyConditionalPrior(0.0, 200.0),
        PriorType::log_uniform)
{

}

void MyModel::from_prior(RNG& rng)
{

}

double MyModel::perturb(RNG& rng)
{
	double logH = 0.;

	return logH;
}

double MyModel::log_likelihood() const
{
	double logL = 0.;
	return logL;
}

void MyModel::print(std::ostream& out) const
{

}

string MyModel::description() const
{
	return string("");
}


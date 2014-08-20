#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:objects(3, 10, false, MyDistribution(-10., 10.))
,C(Data::get_instance().get_t().size(),
	vector<long double>(Data::get_instance().get_t().size(), 0.))
{

}

void MyModel::fromPrior()
{
	objects.fromPrior();
	objects.consolidate_diff();
	calculate_C();
}

void MyModel::calculate_C()
{
	// Get the times from the data
	const vector<double>& t = Data::get_instance().get_t();

	// Update or from scratch?
	bool update = (objects.get_added().size() < objects.get_components().size());

	// Get the components
	const vector< vector<double> >& components = (update)?(objects.get_added()):
				(objects.get_components());

	// Zero the signal
	if(!update)
		C.assign(C.size(), vector<long double>(C[0].size(), 0.));
}

double MyModel::perturb()
{
	double logH = 0.;

	logH += objects.perturb();
	objects.consolidate_diff();
	calculate_C();

	return logH;
}

double MyModel::logLikelihood() const
{
	// Get the data
	const vector<double>& y = Data::get_instance().get_y();

	double logL = 0.;

	return logL;
}

void MyModel::print(std::ostream& out) const
{
	objects.print(out); out<<' ';
}

string MyModel::description() const
{
	return string("objects");
}


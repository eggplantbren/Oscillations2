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

	// Calculate frequencies from log periods
	vector<double> frequencies(components.size());
	for(size_t i=0; i<components.size(); i++)
		frequencies[i] = exp(-components[i][0]);

	// Calculate squared amplitudes
	vector<double> A2(components.size());
	for(size_t i=0; i<A2.size(); i++)
		A2[i] = pow(components[i][1], 2);

	// Calculate 1/(mode lifetimes)
	vector<double> g(components.size());
	for(size_t i=0; i<g.size(); i++)
		g[i] = 1./(exp(components[i][0])*components[i][2]);

	double dt;
	double c;
	for(size_t i=0; i<C.size(); i++)
	{
		for(size_t j=i; j<C[i].size(); j++)
		{
			c = 0.;
			dt = t[i] - t[j];
			for(size_t k=0; k<components.size(); k++)
			{
				c += A2[k]
					*exp(-abs(dt)*g[k])
					*cos(2*M_PI*dt*frequencies[k]);
			}
		}
	}

	/* NO PROMISES THIS WILL ALWAYS BE POSITIVE DEFINITE! */
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


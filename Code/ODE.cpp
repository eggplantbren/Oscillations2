#include "ODE.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>

using namespace std;
using namespace DNest3;

ODE::ODE()
:objects(3, 30, false, MyDistribution())
{

}

void ODE::fromPrior()
{
	objects.fromPrior();
	extra_sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
}


double ODE::perturb()
{
	double logH = 0.;

	logH += objects.perturb();

	extra_sigma = log(extra_sigma);
	extra_sigma = (atan(extra_sigma)/M_PI + 0.485)/0.97;
	extra_sigma += randh();
	wrap(extra_sigma, 0., 1.);
	extra_sigma = tan(M_PI*(0.97*extra_sigma - 0.485));
	extra_sigma = exp(extra_sigma);

	return logH;
}

double ODE::logLikelihood() const
{
	double logL = 0.;
	return logL;
}

void ODE::print(std::ostream& out) const
{
	objects.print(out);
	out<<extra_sigma<<' ';
}

string ODE::description() const
{
	return string("objects");
}

void ODE::advance_RK4(vector<double>& state, double dt)
{
	vector<double> f1 = deriv(state);

	vector<double> temp = state;
	for(size_t i=0; i<temp.size(); i++)
		temp[i] += 0.5*dt*f1[i];
	vector<double> f2 = deriv(temp);

	temp = state;
	for(size_t i=0; i<temp.size(); i++)
		temp[i] += 0.5*dt*f2[i];
	vector<double> f3 = deriv(temp);

	temp = state;
	for(size_t i=0; i<temp.size(); i++)
		temp[i] += dt*f3[i];
	vector<double> f4 = deriv(temp);

	double C = dt/6.;
	for(size_t i=0; i<state.size(); i++)
		state[i] += C*(f1[i] + 2*f2[i] + 2*f3[i] + f4[i]);
}

vector<double> ODE::deriv(const std::vector<double>& state)
{
	vector<double> d(state.size());
	d[0] =  state[1];
	d[1] = -state[0] + 0.1*state[1];
	return d;
}

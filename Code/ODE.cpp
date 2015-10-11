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


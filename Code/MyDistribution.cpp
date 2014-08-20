#include "MyDistribution.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace DNest3;

MyDistribution::MyDistribution(double x_min, double x_max)
:x_min(x_min)
,x_max(x_max)
{

}

void MyDistribution::fromPrior()
{
	mu = exp(tan(M_PI*(0.97*randomU() - 0.485)));
	b = exp(log(1.) + log(1E3)*randomU());
}

double MyDistribution::perturb_parameters()
{
	double logH = 0.;

	int which = randInt(2);

	if(which == 0)
	{
		mu = log(mu);
		mu = (atan(mu)/M_PI + 0.485)/0.97;
		mu += randh();
		wrap(mu, 0., 1.);
		mu = tan(M_PI*(0.97*mu - 0.485));
		mu = exp(mu);
	}
	else
	{
		b = log(b);
		b += log(1E3)*randh();
		wrap(b, log(1.), log(1E3));
		b = exp(b);
	}

	return logH;
}

// vec[0] = "position" (log-period)
// vec[1] = amplitude
// vec[2] = k (mode lifetime in units of periods)

double MyDistribution::log_pdf(const std::vector<double>& vec) const
{
	if(vec[0] < x_min || vec[0] > x_max || vec[1] < 0. ||
			vec[2] < 0. || vec[2] > b)
		return -1E300;

	return -log(mu) - vec[1]/mu - log(b);
}

void MyDistribution::from_uniform(std::vector<double>& vec) const
{
	vec[0] = x_min + (x_max - x_min)*vec[0];
	vec[1] = -mu*log(1. - vec[1]);
	vec[2] = b*vec[2];
}

void MyDistribution::to_uniform(std::vector<double>& vec) const
{
	vec[0] = (vec[0] - x_min)/(x_max - x_min);
	vec[1] = 1. - exp(-vec[1]/mu);
	vec[2] = vec[2]/b;
}

void MyDistribution::print(std::ostream& out) const
{
	out<<mu<<' '<<b<<' ';
}


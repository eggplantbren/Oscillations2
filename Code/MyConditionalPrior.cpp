#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include "DNest4/code/Distributions/Cauchy.h"
#include <cmath>

using namespace DNest4;

MyConditionalPrior::MyConditionalPrior(double f_min, double f_max)
:f_min(f_min)
,f_max(f_max)
{

}

void MyConditionalPrior::from_prior(RNG& rng)
{
    Cauchy c;
    A_min = exp(c.generate(rng));
    mu = 3.0*rng.rand();
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.0;

    int which = rng.rand_int(2);
    if(which == 0)
    {
        Cauchy c;
        A_min = log(A_min);
        logH += c.perturb(A_min, rng);
        A_min = exp(A_min);
    }
    else
    {
        mu += 3.0*rng.randh();
        wrap(mu, 0.0, 3.0);
    }

	return logH;
}

// vec is {frequency, amplitude}
double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    // Hard limits
    if(vec[0] < f_min || vec[0] > f_max)
        return -std::numeric_limits<double>::max();
    if(vec[1] < A_min)
        return -std::numeric_limits<double>::max();
    return -log(mu*vec[1]) - (vec[1]/A_min)/mu;
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    vec[0] = f_min + (f_max - f_min)*vec[0];
    vec[1] = A_min*exp(-mu*log(1.0 - vec[1]));
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = (vec[0] - f_min)/(f_max - f_min);
    vec[1] = 1.0 - exp(-log(vec[1]/A_min)/mu);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	out<<A_min<<' '<<mu<<' ';
}


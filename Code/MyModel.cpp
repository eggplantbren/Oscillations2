#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include "DNest4/code/Distributions/Cauchy.h"
#include "Data.h"
#include <Eigen/Cholesky>

using namespace std;
using namespace DNest4;

const Data& MyModel::data = Data::get_instance();

MyModel::MyModel()
:modes(2, 100, false,
        MyConditionalPrior(0.0, 200.0),
        PriorType::log_uniform)
,C(data.get_y().size(), data.get_y().size())
{

}

void MyModel::from_prior(RNG& rng)
{
    modes.from_prior(rng);

    Cauchy c;
    mode_lifetime = exp(c.generate(rng));

    calculate_C();
}

double MyModel::perturb(RNG& rng)
{
	double logH = 0.0;

    int which = rng.rand_int(2);
    if(which == 0)
    {
        logH += modes.perturb(rng);
    }
    else
    {
        Cauchy c;
        mode_lifetime = log(mode_lifetime);
        logH += c.perturb(mode_lifetime, rng);
        mode_lifetime = exp(mode_lifetime);
    }

    calculate_C();
	return logH;
}

void MyModel::calculate_C()
{
	// Get the times and noise variances from the data
	const vector<double>& t = data.get_t();
	const vector<double>& sig = data.get_sig();

	// Get the components
	const auto& components = modes.get_components();

	// Zero
    for(int i=0; i<C.size(); ++i)
        for(int j=0; j<C.size(); ++j)
            C(i, j) = 0.0;

    // Error bars
    for(int i=0; i<C.size(); ++i)
        C(i, i) += pow(sig[i], 2);

    // Extract frequencies and amplitudes
    vector<double> A(components.size());
    vector<double> f(components.size());
    for(size_t i=0; i<components.size(); ++i)
    {
        f[i] = components[i][0];
        A[i] = components[i][1];
    }

    double dt;
    for(int i=0; i<C.size(); ++i)
    {
        for(int j=(i+1); j<C.size(); ++j)
        {
            dt = std::abs(t[i] - t[j]);

            for(size_t k=0; k<components.size(); ++k)
            {                
                C(i, j) += pow(A[k], 2)*cos(2*M_PI*f[k]*dt)
                                                        *exp(-dt/mode_lifetime);
            }
            C(j, i) = C(i, j);
        }
    }

	/* NO PROMISES THIS WILL ALWAYS BE POSITIVE DEFINITE! */
}

double MyModel::log_likelihood() const
{
	// Get the data
	const vector<double>& d = data.get_y();

	// Copy it into an Eigen vector
	Eigen::VectorXd y(d.size());
	for(int i=0; i<y.size(); i++)
		y(i) = d[i];

	// Cholesky Decomp
	Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();

	Eigen::MatrixXd L = cholesky.matrixL();
	double logDeterminant = 0.;
	for(int i=0; i<y.size(); i++)
		logDeterminant += 2.*log(L(i,i));

	// C^-1*(y-mu)
	Eigen::VectorXd solution = cholesky.solve(y);

	// y*solution
	double exponent = 0.;
	for(int i=0; i<y.size(); i++)
		exponent += y(i)*solution(i);

	double logL = -0.5*y.size()*log(2*M_PI)
			- 0.5*logDeterminant - 0.5*exponent;

	if(isnan(logL) || isinf(logL))
		logL = -1E300;

	return logL;
}

void MyModel::print(std::ostream& out) const
{
    modes.print(out);
    out<<mode_lifetime;
}

string MyModel::description() const
{
	return string("");
}


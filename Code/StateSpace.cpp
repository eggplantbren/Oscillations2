#include "StateSpace.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace DNest3;
using namespace Eigen;

StateSpace::StateSpace()
:objects(3, 1, true, MyDistribution())
{

}

void StateSpace::fromPrior()
{
	objects.fromPrior();
	extra_sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
}

double StateSpace::perturb()
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

double StateSpace::logLikelihood() const
{
	double logL = 0.;

	// Get the data
	const VectorXd& Y = Data::get_instance().get_y_eigen();

	// Get the modes
	const vector< vector<double> >& components = objects.get_components();

	// Mode parameters
	double omega0 = 2*M_PI*exp(-components[0][0]);
	double A = components[0][1];
	double tau = components[0][2]*exp(components[0][0]);
	double D = A*A*omega0*omega0*tau;
	double omega = sqrt(omega0*omega0 - 1.0/(4*tau*tau));

	// The matrices M, I, and J
	MatrixXd M(2, 2), I(2, 2), J(2, 2);
	M<<0.0, -1.0, omega0*omega0, 1.0/tau;
	I<<1.0, 0.0, 0.0, 1.0;
	J<<1.0/(2*omega*tau), 1.0/omega, -omega0*omega0/omega, -1.0/(2*omega*tau);

	// Probability distribution for the signal (x, v)
	// is gaussian. Its mean and covariance matrix are mu and C
	VectorXd mu = VectorXd::Zero(2);
	MatrixXd C(2, 2);

	// For the first point, Dt=Infinity
	C(0, 0) = D/(4*pow(omega*omega0, 2)*pow(tau, 3))*(4*pow(omega*tau, 2));
	C(0, 1) = 0.;
	C(1, 0) = 0.;
	C(1, 1) = D/(4*pow(omega, 2)*pow(tau, 3))*4*pow(omega*tau, 2);

	double var;
	for(int i=0; i<Y.size(); i++)
	{
		// Probability distribution for the data point given the signal
		var = C(0, 0) + pow(extra_sigma, 2);
		logL += -0.5*log(2*M_PI*var) - 0.5*pow(Y[i], 2)/var;
		

	}

	return logL;
}

void StateSpace::print(std::ostream& out) const
{
	objects.print(out);
	out<<extra_sigma<<' ';
}

string StateSpace::description() const
{
	return string("objects");
}


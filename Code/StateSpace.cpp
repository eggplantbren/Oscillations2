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
	const vector<double>& t = Data::get_instance().get_t();
	const VectorXd& Y = Data::get_instance().get_y_eigen();
	const vector<double>& sig = Data::get_instance().get_sig();

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

	// State of knowledge of signal
	VectorXd mu = VectorXd::Zero(2);
	MatrixXd C(2, 2);
	C(0, 0) = D/pow(omega0, 2)/tau;
	C(0, 1) = 0.;
	C(1, 0) = 0.;
	C(1, 1) = D/tau;

	// Declare stuff
	double Dt, var;
	MatrixXd C1inv(2, 2), C2inv(2, 2), C3(2, 2), mexp(2, 2);
	VectorXd mu2(2);

	for(int i=0; i<Y.size(); i++)
	{
		// Evaluate probability distribution for the data point
		var = C(0, 0) + pow(sig[i], 2);//pow(extra_sigma, 2);
		logL += -0.5*log(2*M_PI*var) - 0.5*pow(Y[i] - mu[0], 2)/var;

		// Update knowledge of signal at current time
		C1inv = C.inverse();
		C2inv<<(1.0/pow(sig[i], 2)), 0.0, 0.0, 0.0;
		C3 = (C1inv + C2inv).inverse();
		mu2<<Y[i], 0.;
		mu = C3*C1inv*mu + C3*C2inv*mu2;
		C = C3;

		// Calculate knowledge of signal at next time
		if(i != Y.size() - 1)
		{
			Dt = t[i+1] - t[i];

			// Just reuse C3, don't need a new matrix
			C3(0, 0) = D/(4*pow(omega*omega0, 2)*pow(tau, 3))*
						(4*pow(omega*tau, 2) + exp(-Dt/tau)*
						(cos(2*omega*Dt) - 2*omega*tau*sin(2*omega*Dt) -
								 4*pow(omega0*tau, 2)));
			C3(0, 1) = D/pow(omega, 2)/pow(tau, 2)*exp(-Dt/tau)*pow(sin(omega*Dt), 2);
			C3(1, 0) = C3(0, 1);
			C3(1, 1) = D/(4*pow(omega, 2)*pow(tau, 3))*
						(4*pow(omega*tau, 2) + exp(-Dt/tau)*
						(cos(2*omega*Dt) + 2*omega*tau*sin(2*omega*Dt) - 4*pow(omega0*tau, 2)));

			mexp = exp(-0.5*Dt/tau)*(cos(omega*Dt)*I + sin(omega*Dt)*J);
			mu = mexp*mu;
			C = mexp*C*mexp.transpose() + C3;
		}

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


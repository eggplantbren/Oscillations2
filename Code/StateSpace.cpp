#include "StateSpace.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

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
	int N = components.size();

	// Mode parameters
	vector<double> omega0(N);
	vector<double> A(N);
	vector<double> tau(N);
	vector<double> D(N);
	vector<double> omega(N);
	for(int i=0; i<N; i++)
	{
		omega0[i] = 2*M_PI*exp(-components[i][0]);
		A[i] = components[i][1];
		tau[i] = components[i][2]*exp(components[i][0]);
		D[i] = A[i]*A[i]*omega0[i]*omega0[i]*tau[i];
		omega[i] = sqrt(omega0[i]*omega0[i] - 1.0/(4*tau[i]*tau[i]));
	}

	// State of knowledge of signal
	// (y1, v1, y2, v2, ..., yN, vN)
	VectorXd mu = VectorXd::Zero(2*N);
	MatrixXd C = MatrixXd::Zero(2*N, 2*N);

	// The matrices M, I, and J
	MatrixXd M = MatrixXd::Zero(2*N, 2*N);
	MatrixXd I = MatrixXd::Identity(2*N, 2*N);
	MatrixXd J = MatrixXd::Zero(2*N, 2*N);
	for(int i=0; i<N; i++)
	{
		M(2*i, 2*i+1) = -1.;
		M(2*i+1, 2*i) = pow(omega0[i], 2);
		M(2*i+1, 2*i+1) = 1./tau[i];

		J(2*i, 2*i) = 1./(2*omega[i]*tau[i]);
		J(2*i, 2*i+1) = -1./omega[i];
		J(2*i+1, 2*i) = -pow(omega0[i], 2)/omega[i];
		J(2*i+1, 2*i+1) = -1./(2*omega[i]*tau[i]);

		C(2*i, 2*i) = D[i]/pow(omega0[i], 2)/tau[i];
		C(2*i+1, 2*i+1) = D[i]/tau[i];
	}

	// Declare stuff
	double Dt, mean, var;
	MatrixXd C1inv(2*N, 2*N), C2inv(2*N, 2*N), C3(2*N, 2*N), mexp(2*N, 2*N);
	VectorXd mu2(2*N);

	for(int i=0; i<Y.size(); i++)
	{
		// Evaluate probability distribution for the data point
		mean = 0.;
		var = 0.;
		for(int j=0; j<N; j++)
		{
			mean += mu(2*j);
			var += C(2*j, 2*j);
		}
		var += pow(sig[i], 2);
		logL += -0.5*log(2*M_PI*var) - 0.5*pow(Y[i] - mean, 2)/var;

		// Update knowledge of signal at current time
		C1inv = C.inverse();
		C2inv = MatrixXd::Zero(2*N, 2*N);
		for(int j=0; j<N; j++)
			C2inv(2*j, 2*j) = pow(sig[i], -2);
		C3 = (C1inv + C2inv).inverse();
		mu2 = VectorXd::Ones(2*N);
		for(int j=0; j<N; j++)
			mu2(2*j) = Y[i]/N;
		mu = C3*C1inv*mu + C3*C2inv*mu2;
		C = C3;

		// Calculate knowledge of signal at next time
		if(i != Y.size() - 1)
		{
			Dt = t[i+1] - t[i];

			// Just reuse C3, don't need a new matrix
			C3 = MatrixXd::Zero(2*N, 2*N);
			for(int j=0; j<N; j++)
			{
				C3(2*j, 2*j) = D[j]/(4*pow(omega[j]*omega0[j], 2)*pow(tau[j], 3))*
							(4*pow(omega[j]*tau[j], 2) + exp(-Dt/tau[j])*
							(cos(2*omega[j]*Dt) - 2*omega[j]*tau[j]*sin(2*omega[j]*Dt) -
									 4*pow(omega0[j]*tau[j], 2)));
				C3(2*j, 2*j+1) = D[j]/pow(omega[j], 2)/pow(tau[j], 2)*exp(-Dt/tau[j])*pow(sin(omega[j]*Dt), 2);
				C3(2*j+1, 2*j) = C3(0, 1);
				C3(2*j+1, 2*j+1) = D[j]/(4*pow(omega[j], 2)*pow(tau[j], 3))*
							(4*pow(omega[j]*tau[j], 2) + exp(-Dt/tau[j])*
							(cos(2*omega[j]*Dt) + 2*omega[j]*tau[j]*sin(2*omega[j]*Dt) - 4*pow(omega0[j]*tau[j], 2)));
			}

			mexp = (-Dt*M).exp();
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


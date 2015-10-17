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

	// Some coefficients
	vector<double> c1(N), c2(N), c3(N), c4(N), c5(N);
	for(int i=0; i<N; i++)
	{
		omega0[i] = 2*M_PI*exp(-components[i][0]);
		A[i] = components[i][1];
		tau[i] = components[i][2]*exp(components[i][0]);
		D[i] = A[i]*A[i]*omega0[i]*omega0[i]*tau[i];
		omega[i] = sqrt(omega0[i]*omega0[i] - 1.0/(4*tau[i]*tau[i]));

		c1[i] = D[i]/(4*pow(omega[i]*omega0[i], 2)*pow(tau[i], 3));
		c2[i] = D[i]/pow(omega[i], 2)/pow(tau[i], 2);
		c3[i] = D[i]/(4*pow(omega[i], 2)*pow(tau[i], 3));
		c4[i] = 4*pow(omega[i]*tau[i], 2);
		c5[i] = 4*pow(omega0[i]*tau[i], 2);
	}

	// State of knowledge of signal
	// (y1, v1, y2, v2, ..., yN, vN)
	VectorXd mu = VectorXd::Zero(2*N);
	MatrixXd C = MatrixXd::Zero(2*N, 2*N);

	// The matrix M
	MatrixXd M = MatrixXd::Zero(2*N, 2*N);
	for(int i=0; i<N; i++)
	{
		M(2*i, 2*i+1) = -1.;
		M(2*i+1, 2*i) = pow(omega0[i], 2);
		M(2*i+1, 2*i+1) = 1./tau[i];

		C(2*i, 2*i) = D[i]/pow(omega0[i], 2)/tau[i];
		C(2*i+1, 2*i+1) = D[i]/tau[i];
	}

	// Declare stuff
	double Dt, mean, var, junk1, junk2;
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
		junk1 = 1./pow(sig[i], 2);
		junk2 = Y[i]/N/pow(sig[i], 2);
		C1inv = C.inverse();
		C2inv = MatrixXd::Zero(2*N, 2*N);
		for(int j=0; j<N; j++)
			C2inv(2*j, 2*j) = junk1;
		C3 = (C1inv + C2inv).inverse();
		mu2 = VectorXd::Zero(2*N);
		for(int j=0; j<N; j++)
			mu2(2*j) = junk2;
		mu = C3*C1inv*mu + C3*mu2;
		C = C3;

		// Calculate knowledge of signal at next time
		if(i != Y.size() - 1)
		{
			Dt = t[i+1] - t[i];

			// Just reuse C3, don't need a new matrix
			C3 = MatrixXd::Zero(2*N, 2*N);
			for(int j=0; j<N; j++)
			{
				C3(2*j, 2*j) = c1[j]*
							(c4[j] + exp(-Dt/tau[j])*
							(cos(2*omega[j]*Dt) - 2*omega[j]*tau[j]*sin(2*omega[j]*Dt) - c5[j]));
				C3(2*j, 2*j+1) = c2[j]*exp(-Dt/tau[j])*pow(sin(omega[j]*Dt), 2);
				C3(2*j+1, 2*j) = C3(0, 1);
				C3(2*j+1, 2*j+1) = c3[j]*
							(c4[j] + exp(-Dt/tau[j])*
							(cos(2*omega[j]*Dt) + 2*omega[j]*tau[j]*sin(2*omega[j]*Dt) - c5[j]));
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


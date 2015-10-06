#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Cholesky>


// Eigen is column major and numpy is row major. Barf.
typedef Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
						  Eigen::RowMajor> > RowMajorMap;

using namespace std;
using namespace DNest3;
using namespace Eigen;

MyModel::MyModel()
:objects(3, 30, false, MyDistribution())
{

}

void MyModel::fromPrior()
{
	objects.fromPrior();
	objects.consolidate_diff();
	extra_sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
}

double MyModel::calculate_C(int i, int j) const
{
	// Get the times and noise variances from the data
	const vector<double>& t = Data::get_instance().get_t();
	const vector<double>& sig = Data::get_instance().get_sig();

	// Get the components
	const vector< vector<double> >& components = objects.get_components();

	double C = 0.;

	// Add diagonal part for noise in the measurements
	if(i == j)
		C += pow(sig[i], 2) + pow(extra_sigma, 2);

	// Calculate frequencies from log periods
	vector<double> frequencies(components.size());
	for(size_t k=0; k<components.size(); k++)
		frequencies[k] = exp(-components[k][0]);

	// Calculate squared amplitudes
	vector<double> A2(components.size());
	for(size_t k=0; k<A2.size(); k++)
		A2[k] = pow(components[k][1], 2);

	// Calculate 1/(mode lifetimes)
	vector<double> g(components.size());
	for(size_t k=0; k<g.size(); k++)
		g[k] = 1./(exp(components[k][0])*components[k][2]);

	double dt;
	dt = t[i] - t[j];
	for(size_t k=0; k<components.size(); k++)
		C += A2[k]*exp(-abs(dt)*g[k])*cos(2*M_PI*dt*frequencies[k]);
	/* NO PROMISES THIS WILL ALWAYS BE POSITIVE DEFINITE! */

	return C;
}

double MyModel::perturb()
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

	void apply_inverse (const unsigned int n, const unsigned int nrhs,
						double* b, double* out) {
	};

double MyModel::logLikelihood() const
{
	// Get the data
	const VectorXd& y = Data::get_instance().get_y_eigen();

	// Compute the diagonal elements.
	VectorXd diag(y.size());
	for(size_t i=0; i<y.size(); i++)
		diag[i] = calculate_C(i, i);

	HODLRSolverMatrix matrix(*this);
	HODLR_Tree<HODLRSolverMatrix>* solver = new HODLR_Tree<HODLRSolverMatrix> (&matrix, y.size(), 30);

	solver->assemble_Matrix(diag, 1E-10, 's');

	// Factorize the matrix.
	solver->compute_Factor();

	double logdet;

	// Extract the log-determinant.
	solver->compute_Determinant(logdet);

	double* b = new double[y.size()];
	double* out = new double[y.size()];
	for(int i=0; i<y.size(); i++)
		b[i] = y[i];

	MatrixXd b_vec = RowMajorMap(b, y.size(), 1);
	MatrixXd alpha(y.size(), 1);
	solver->solve(b_vec, alpha);
	for(int i=0; i<y.size(); i++)
		out[i] = alpha(i, 0);

	double exponent = 0.;
	for(int i=0; i<y.size(); i++)
		exponent += y(i)*out[i];

	double logL = -0.5*y.size()*log(2*M_PI) - 0.5*logdet - 0.5*exponent;

	if(isnan(logL) || isinf(logL))
		logL = -1E300;

	delete[] b;
	delete[] out;
	delete solver;
	return logL;
}

void MyModel::print(std::ostream& out) const
{
	objects.print(out);
	out<<extra_sigma<<' ';
}

string MyModel::description() const
{
	return string("objects");
}


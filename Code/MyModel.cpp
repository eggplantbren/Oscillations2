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
,C(Data::get_instance().get_t().size(),
	vector<long double>(Data::get_instance().get_t().size(), 0.))
{

}

void MyModel::fromPrior()
{
	objects.fromPrior();
	objects.consolidate_diff();
	extra_sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));

	calculate_C();
}

void MyModel::calculate_C()
{
	// Get the times and noise variances from the data
	const vector<double>& t = Data::get_instance().get_t();
	const vector<double>& sig = Data::get_instance().get_sig();

	// Update or from scratch?
	bool update = (objects.get_added().size() < objects.get_components().size());
	update = update && (staleness < 10);

	// Get the components
	const vector< vector<double> >& components = (update)?(objects.get_added()):
				(objects.get_components());

	// Zero the signal
	if(!update)
	{
		C.assign(C.size(), vector<long double>(C[0].size(), 0.));

		// Add diagonal part for noise in the measurements
		for(size_t i=0; i<C.size(); i++)
			C[i][i] = pow(sig[i], 2) + pow(extra_sigma, 2);

		staleness = 0;
	}
	else
		staleness++;

	// Calculate frequencies from log periods
	vector<double> frequencies(components.size());
	for(size_t i=0; i<components.size(); i++)
		frequencies[i] = exp(-components[i][0]);

	// Calculate squared amplitudes
	vector<double> A2(components.size());
	for(size_t i=0; i<A2.size(); i++)
	{
		A2[i] = pow(components[i][1], 2);
		if(components[i][1] < 0.)
			A2[i] *= -1;
	}

	// Calculate 1/(mode lifetimes)
	vector<double> g(components.size());
	for(size_t i=0; i<g.size(); i++)
		g[i] = 1./(exp(components[i][0])*components[i][2]);

	double dt;
	double c;
	for(size_t i=0; i<C.size(); i++)
	{
		for(size_t j=i; j<C[i].size(); j++)
		{
			c = 0.;
			dt = t[i] - t[j];
			for(size_t k=0; k<components.size(); k++)
			{
				c += A2[k]
					*exp(-abs(dt)*g[k])
					*cos(2*M_PI*dt*frequencies[k]);
			}

			C[i][j] += c;
			if(i != j)
				C[j][i] += c;
		}
	}

	/* NO PROMISES THIS WILL ALWAYS BE POSITIVE DEFINITE! */
}

double MyModel::perturb()
{
	double logH = 0.;

	logH += objects.perturb();
	objects.consolidate_diff();

	// Update extra noise parameter
	double diff = -extra_sigma*extra_sigma;
	extra_sigma = log(extra_sigma);
	extra_sigma = (atan(extra_sigma)/M_PI + 0.485)/0.97;
	extra_sigma += randh();
	wrap(extra_sigma, 0., 1.);
	extra_sigma = tan(M_PI*(0.97*extra_sigma - 0.485));
	extra_sigma = exp(extra_sigma);
	diff += extra_sigma*extra_sigma;
	// Update C based on extra noise move
	for(size_t i=0; i<C.size(); i++)
		C[i][i] += diff;

	calculate_C();

	return logH;
}

    void apply_inverse (const unsigned int n, const unsigned int nrhs,
                        double* b, double* out) {
    };

double MyModel::logLikelihood() const
{
	// Get the data
	const vector<double>& d = Data::get_instance().get_y();
	// Copy it into an Eigen vector
	VectorXd y(d.size());
	for(int i=0; i<y.size(); i++)
		y(i) = d[i];

	// Copy covariance matrix into an Eigen matrix
	MatrixXd CC(C.size(), C[0].size());
	for(size_t i=0; i<C.size(); i++)
		for(size_t j=0; j<C[i].size(); j++)
			CC(i, j) = C[i][j];
        // Compute the diagonal elements.
        VectorXd diag(C.size());
        for(size_t i=0; i<C.size(); i++)
		diag[i] = C[i][i];

	HODLRSolverMatrix matrix(*this);
	HODLR_Tree<HODLRSolverMatrix>* solver = new HODLR_Tree<HODLRSolverMatrix> (&matrix, C.size(), 30);

        solver->assemble_Matrix(diag, 1E-10, 's');

        // Factorize the matrix.
        solver->compute_Factor();

	double logdet;
        // Extract the log-determinant.
        solver->compute_Determinant(logdet);

        double* b = new double[C.size()];
	double* out = new double[C.size()];
	for(size_t i=0; i<C.size(); i++)
		b[i] = y[i];

        MatrixXd b_vec = RowMajorMap(b, C.size(), 1), alpha(C.size(), 1);
        solver->solve(b_vec, alpha);
        for(size_t i = 0; i<C.size(); ++i)
          out[i] = alpha(i, 0);

//	// Cholesky Decomp
//	Eigen::LLT<Eigen::MatrixXd> cholesky = CC.llt();

//	MatrixXd L = cholesky.matrixL();
//	double logDeterminant = 0.;
//	for(int i=0; i<y.size(); i++)
//		logDeterminant += 2.*log(L(i,i));

//	// C^-1*(y-mu)
//	VectorXd solution = cholesky.solve(y);

//	cout<<logdet<<' '<<logDeterminant<<endl;
//	cout<<out[0]<<' '<<solution[0]<<endl<<endl;
	//cout<<soln(0,0)<<' '<<solution[0]<<endl<<endl;
	delete[] b;
	delete[] out;

	// y*solution
	double exponent = 0.;
	for(int i=0; i<y.size(); i++)
		exponent += y(i)*out[i];

	double logL = -0.5*y.size()*log(2*M_PI)
			- 0.5*logdet - 0.5*exponent;

	if(isnan(logL) || isinf(logL))
		logL = -1E300;

	delete solver;
	return logL;

}

void MyModel::print(std::ostream& out) const
{
	objects.print(out);
	out<<extra_sigma<<' ';
	out<<' '<<staleness<<' ';
}

string MyModel::description() const
{
	return string("objects");
}


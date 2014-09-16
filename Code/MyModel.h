#ifndef _MyModel_
#define _MyModel_

#include "Model.h"
#include <vector>
#include <RJObject.h>
#include "MyDistribution.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>

class MyModel:public DNest3::Model
{
	private:
		RJObject<MyDistribution> objects;

		double extra_sigma;

		// The covariance matrix
		std::vector< std::vector<long double> > C;

		// Count updates without full recalculation of C
		unsigned int staleness;

		void calculate_C();

	public:
		MyModel();

		// Generate the point from the prior
		void fromPrior();

		// Metropolis-Hastings proposals
		double perturb();

		// Likelihood function
		double logLikelihood() const;

		// Print to stream
		void print(std::ostream& out) const;

		// Return string with column information
		std::string description() const;
};

#endif


#ifndef _StateSpace_
#define _StateSpace_

#include "Model.h"
#include <vector>
#include <RJObject.h>
#include "MyDistribution.h"
#include <Eigen/Dense>

class StateSpace:public DNest3::Model
{
	private:
		RJObject<MyDistribution> objects;
		double extra_sigma;

	public:
		StateSpace();

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


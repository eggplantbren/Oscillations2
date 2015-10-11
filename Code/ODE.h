#ifndef _ODE_
#define _ODE_

#include "Model.h"
#include <vector>
#include <RJObject.h>
#include "MyDistribution.h"

class ODE:public DNest3::Model
{

	private:
		RJObject<MyDistribution> objects;
		double extra_sigma;

		double calculate_C(int i, int j) const;

	public:
		ODE();

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


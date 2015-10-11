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

		// Calculate derivatives
		std::vector<double> deriv(const std::vector<double>& state);

		// RK4 algorithm
		void advance_RK4(std::vector<double>& state, double dt);

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


#ifndef Oscillations2_MyModel
#define Oscillations2_MyModel

#include "DNest4/code/DNest4.h"
#include "MyConditionalPrior.h"
#include "Data.h"
#include <ostream>
#include <Eigen/Dense>

class MyModel
{
	private:
        DNest4::RJObject<MyConditionalPrior> modes;
        double mode_lifetime;

        // Covariance matrix
        Eigen::MatrixXd C;
        void calculate_C();

        static const Data& data;

	public:
		// Constructor only gives size of params
		MyModel();

		// Generate the point from the prior
		void from_prior(DNest4::RNG& rng);

		// Metropolis-Hastings proposals
		double perturb(DNest4::RNG& rng);

		// Likelihood function
		double log_likelihood() const;

		// Print to stream
		void print(std::ostream& out) const;

		// Return string with column information
		std::string description() const;
};

#endif


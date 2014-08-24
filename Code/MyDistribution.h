#ifndef _MyDistribution_
#define _MyDistribution_

#include <Distributions/Distribution.h>

// Based on ClassicMassInf1D from RJObject
// Think of "position x" as log-period
// and mass as amplitude
class MyDistribution:public Distribution
{
	private:
		// Center and width of bi-exponential prior for log periods
		double center, width;

		// Mean of exponential interim prior for amplitudes
		double mu;

		// Upper limit of uniform interim prior for
		// k = (mode lifetime)/period
		double b;

		double perturb_parameters();

	public:
		MyDistribution();

		void fromPrior();

		double log_pdf(const std::vector<double>& vec) const;
		void from_uniform(std::vector<double>& vec) const;
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};

#endif


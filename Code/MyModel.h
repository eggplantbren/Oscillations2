#ifndef _MyModel_
#define _MyModel_

#include "Model.h"
#include <vector>
#include <RJObject.h>
#include "MyDistribution.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <HODLR_Matrix.hpp>
#include <HODLR_Tree.hpp>

class MyModel:public DNest3::Model
{
	friend class HODLRSolverMatrix;

	private:
		RJObject<MyDistribution> objects;

		double extra_sigma;

		double calculate_C(int i, int j) const;

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


class HODLRSolverMatrix : public HODLR_Matrix
{
	private:
		const MyModel& model;

public:
    HODLRSolverMatrix (const MyModel& model)
        : model(model)
    {
    }

    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return model.calculate_C(i, j);
    };

};


#endif


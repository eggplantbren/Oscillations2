#ifndef _NotPeriodogram_
#define _NotPeriodogram_

#include "Model.h"
#include <vector>
#include <RJObject.h>
#include "MyDistribution.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <HODLR_Matrix.hpp>
#include <HODLR_Tree.hpp>

class NotPeriodogram:public DNest3::Model
{
	friend class HODLRSolverMatrix;

	private:
		// log period, amplitude, (mode lifetime)/period
		double logT, A, K;

		double calculate_C(int i, int j) const;

	public:
		NotPeriodogram(double logT, double A, double K);

		// Likelihood function
		double logLikelihood() const;
};


class HODLRSolverMatrix : public HODLR_Matrix
{
	private:
		const NotPeriodogram& model;

public:
    HODLRSolverMatrix (const NotPeriodogram& model)
        : model(model)
    {
    }

    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return model.calculate_C(i, j);
    };

};


#endif


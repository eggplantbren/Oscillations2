#ifndef Oscillations2_Data
#define Oscillations2_Data

#include <vector>
#include "Eigen/Dense"

class Data
{
	private:
		std::vector<double> t, y, sig;
		Eigen::VectorXd y_eigen;

	public:
		Data();
		void load(const char* filename);

		// Getters
		const std::vector<double>& get_t() const { return t; }
		const std::vector<double>& get_y() const { return y; }
		const Eigen::VectorXd& get_y_eigen() const { return y_eigen; }
		const std::vector<double>& get_sig() const { return sig; }

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif


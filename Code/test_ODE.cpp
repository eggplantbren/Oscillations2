#include <iostream>
#include <vector>
#include "ODE.h"

using namespace std;

int main()
{
	const double dt = 0.01;

	ODE ode;

	vector<double> state(2);
	state[0] = 1.;
	state[1] = 0.;

	for(int i=0; i<10000; i++)
	{
		ode.advance_RK4(state, dt);
		cout<<(i+1)*dt<<' '<<state[0]<<endl;
	}


	return 0;
}


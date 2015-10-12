"""
Code implementing the analytic solution given by
arxiv: 1102.0524
"""

# Angular frequency of oscillator
omega0 = 1.0

# Mode lifetime
tau = 5.0

# Something to do with the strength of the driving force
D = 1.0

# Cyclic frequency of the damped oscillator
omega = sqrt(omega0^2 - 1.0/(4*tau^2))

# The matrices M, I, and J
M = [0.0 -1.0; omega0^2 1.0/tau]
I = eye(2)
J = [1.0/(2*omega*tau) 1.0/omega; -omega0^2/omega -1.0/(2*omega*tau)]

@doc """
Covariance matrix for (delta_x, delta_v) (Equations 15-17)
""" ->
function covariance(Dt::Float64)
	C = Array(Float64, (2, 2))
	C[1, 1] = D/(4*omega^2*omega0^2*tau^3)*
					(4*omega^2*tau^2 + exp(-Dt/tau)*
					(cos(2*omega*Dt) - 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
	C[1, 2] = D/(omega^2*tau^2)*exp(-Dt/tau)*sin(omega*Dt)^2
	C[2, 1] = C[1, 2]
	C[2, 2] = D/(4*omega^2*tau^3)*
					(4*omega^2*tau^2 + exp(-Dt/tau)*
					(cos(2*omega*Dt) + 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
	return C
end

@doc """
Simulate a mode at the input times
""" ->
function simulate(t::Array{Float64, 1})
	y = Array(Float64, length(t))

	# Use stationary distribution for initial conditions
	C = covariance(maximum([1000*tau, 1000*2*pi/omega0]))
	n = randn(2)
	x = C[1, 1]*n[1]
	v = C[1, 2]^2/C[1, 1]*n[1] + sqrt(C[2, 2]^2 - C[1, 2]^4/C[1, 1]^2)*n[2]

	y[1] = x
	for(i in 2:length(t))
		Dt = t[i] - t[i-1]
		C = covariance(Dt)
		n = randn(2)

		# Matrix exponential (Equation 9)
		mexp = exp(-Dt/(2*tau))*(cos(omega*Dt)*I + sin(omega*Dt)*J)
		(x, v) = mexp*[x; v] # Part of Equation 7

		# Equations 13 and 14
		x += C[1, 1]*n[1]
		v += C[1, 2]^2/C[1, 1]*n[1] + sqrt(C[2, 2]^2 - C[1, 2]^4/C[1, 1]^2)*n[2]

		y[i] = x
	end

	return y
end


using PyCall
@pyimport matplotlib.pyplot as plt

t = Array(linspace(0.0, 500.0, 5001))
y = simulate(t)
plt.plot(t, y)
plt.show()


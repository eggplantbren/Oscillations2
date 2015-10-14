"""
Code implementing the analytic solution given by
arxiv: 1102.0524
"""

# Angular frequency of oscillator
omega0 = 2.0*pi/20.0

# Mode lifetime
tau = 60.0

# Amplitude
A = 1.0

# Something to do with the strength of the driving force
D = A^2*omega0^2*tau

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

	if(Dt == Inf)
		C[1, 1] = D/(omega0^2*tau)
		C[1, 2] = 0.0
		C[2, 1] = 0.0
		C[2, 2] = D/tau
	else
		C[1, 1] = D/(4*omega^2*omega0^2*tau^3)*
						(4*omega^2*tau^2 + exp(-Dt/tau)*
						(cos(2*omega*Dt) - 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
		C[1, 2] = D/(omega^2*tau^2)*exp(-Dt/tau)*sin(omega*Dt)^2
		C[2, 1] = C[1, 2]
		C[2, 2] = D/(4*omega^2*tau^3)*
						(4*omega^2*tau^2 + exp(-Dt/tau)*
						(cos(2*omega*Dt) + 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
	end
	return C
end


@doc """
Evolve uncertainty (described by mean and covariance matrix)
for a certain duration
""" ->
function advance!(mu::Vector{Float64}, C::Matrix{Float64}, Dt::Float64)
	# Matrix exponential (Equation 9)
	mexp = exp(-Dt/(2*tau))*(cos(omega*Dt)*I + sin(omega*Dt)*J)
	mu = mexp*mu # Based on Expected value of Equation 7

	# Update covariance matrix
	C = mexp*mexp*C + covariance(dt)
	return nothing
end



@doc """
Simulate a mode at the input times
""" ->
function simulate(t::Array{Float64, 1})
	y = Array(Float64, length(t))

	# Use stationary distribution for initial conditions
	C = covariance(Inf)
	n = randn(2)
	(x, v) = chol(C)'*n
	y[1] = x
	for(i in 2:length(t))
		Dt = t[i] - t[i-1]
		C = covariance(Dt)
		n = randn(2)

		# Matrix exponential (Equation 9)
		mexp = exp(-Dt/(2*tau))*(cos(omega*Dt)*I + sin(omega*Dt)*J)
		(x, v) = mexp*[x; v] # Part of Equation 7

		# Equations 13 and 14
		temp = chol(C)'*n
		x += temp[1]
		v += temp[2]

		y[i] = x
	end

	return y
end


using PyCall
@pyimport matplotlib.pyplot as plt

t = Array(linspace(0.0, 1000.0, 2001))
y = simulate(t)
plt.plot(t, y)
plt.show()


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
function advance(mu::Vector{Float64}, C::Matrix{Float64}, Dt::Float64)
	# Matrix exponential (Equation 9)
	mexp = exp(-Dt/(2*tau))*(cos(omega*Dt)*I + sin(omega*Dt)*J)
	mu = mexp*mu # Based on Expected value of Equation 7

	# Update covariance matrix
	C = mexp*C*mexp' + covariance(Dt)
	return (mu, C)
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

data = readdlm("../data.txt")
plt.errorbar(data[:,1], data[:,2], yerr=data[:,3], fmt="bo")

# Prior state of knowledge about signal
mu = [0.0, 0.0]
C = covariance(Inf)

# Log likelihood
logL = 0.

for(i in 1:size(data)[1])
	# Probability of the data point
	var = C[1, 1] + data[i, 3]
	logL += -0.5*log(2*pi*var) - 0.5*(data[i, 2] - mu[1])^2/var

	plt.errorbar(data[i, 1], mu[1], yerr=sqrt(C[1,1]), fmt="ro")

	# Update knowledge of signal
	# http://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
	C1inv = inv(C)
	C2inv = [1.0/data[i,3]^2 0.0; 0.0 0.0]
	C3 = inv(C1inv + C2inv)
	mu1 = mu
	mu2 = [data[i, 2], 0.0]
	mu3 = C3*C1inv*mu1 + C3*C2inv*mu2
	mu = mu3
	C = C3

	# Evolve
	if(i != size(data)[1])
		(mu, C) = advance(mu, C, data[i+1, 1] - data[i, 1])
	end
end

println(logL)
plt.show()


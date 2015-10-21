
using Optim

"""
Code implementing the analytic solution given by
arxiv: 1102.0524
"""

function log_likelihood(params::Vector{Float64}, data::Matrix{Float64})
	A, tau, nu = params
	omega0 = 2*pi*nu

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




	# Prior state of knowledge about signal
	mu = [0.0, 0.0]
	C = covariance(Inf)

	# Log likelihood
	logL = 0.

	for(i in 1:size(data)[1])
		# Probability of the data point
		var = C[1, 1] + data[i, 3]^2
		logL += -0.5*log(2*pi*var) - 0.5*(data[i, 2] - mu[1])^2/var

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

	return logL
end

function badness(params::Vector{Float64}, data::Matrix{Float64})
	f = 0.0
	try
		f = -log_likelihood(params, data)
	catch
		f = Inf
	end
	return f
end

# Fit mode at the given frequency
function fit_mode(freq::Float64, data::Matrix{Float64})
	function badness2(params::Vector{Float64})
		return badness(vcat(params, freq), data)
	end

	params = [1.0, 10.0]
	result = optimize(badness2, params, method=:cg, ftol=0.01)
	return vcat(result.minimum, result.f_minimum)
end

# Compute the "periodogram" over the given frequency range
function periodogram(freq_min::Float64, freq_max::Float64,
							data::Matrix{Float64}; N::Int64=1000)
	# Frequency spacing
	df = (freq_max - freq_min)/(N - 1)

	freq = Array(Float64, (N, ))
	pgram = Array(Float64, (N, ))
	logl = Array(Float64, (N, ))
	for(i in 1:N)
		freq[i] = freq_min + (i-1)*df
		result = fit_mode(freq[i], data)
		pgram[i] = result[1]^2
		logl[i] = -result[3]
		println("Done ", i, "/", N)
	end

	return (freq, pgram, logl)
end


###############################################
# MAIN PROGRAM
###############################################

using PyCall
@pyimport matplotlib.pyplot as plt

# Load the data and plot the periodogrm
data = readdlm("two_modes.txt")

(freq, pgram, logl) = periodogram(0.5, 3.0, data, N=101)

plt.plot(freq, pgram)
plt.xlabel("Frequency")
plt.show()

plt.plot(freq, logl)
plt.show()



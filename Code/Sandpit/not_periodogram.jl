
using Optim

function fast_inv(A::Matrix{Float64})
	result = [A[2, 2] -A[1, 2]; -A[2, 1] A[1, 1]]/(A[1, 1]*A[2, 2] - A[1, 2]*A[2, 1])
end

function fast_multiplication(A::Matrix{Float64}, b::Matrix{Float64})
	return [A[1,1]*b[1] + A[1,2]*b[2]; A[2,1]*b[1] + A[2,2]*b[2]]
end

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
			o2tau2 = omega^2*tau^2
			decay = exp(-Dt/tau)
			cc = cos(2*omega*Dt)
			ss = 2*omega*tau*sin(2*omega*Dt)
			ff = 4*omega0^2*tau^2
			C[1, 1] = D/(4*o2tau2*omega0^2*tau)*
							(4*o2tau2 + decay*
							(cc - ss - ff))
			C[1, 2] = D/(o2tau2)*decay*sin(omega*Dt)^2
			C[2, 1] = C[1, 2]
			C[2, 2] = D/(4*o2tau2*tau)*
							(4*o2tau2 + decay*
							(cc + ss - ff))
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
		C1inv = fast_inv(C)
		C2inv = [1.0/data[i,3]^2 0.0; 0.0 0.0]
		C3 = fast_inv(C1inv + C2inv)
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

	return logL
end

function badness(params::Vector{Float64}, data::Matrix{Float64})
	f = 0.0

	if(any(params .< 0.0))
		return Inf
	end

	try
		f = -log_likelihood(params, data)
	catch
		f = Inf
	end
	return f
end

# Fit mode at the given frequency
function fit_mode(freq::Float64, data::Matrix{Float64}, A_init::Float64)
	function badness2(params::Vector{Float64})
		return badness(vcat(params, freq), data)
	end

	params = [0.1, 10.0]
	result = optimize(badness2, params, ftol=0.001)
	return vcat(result.minimum, result.f_minimum)
end

# Compute the 'power' at a given frequency
function power(freq::Float64, data::Matrix{Float64})
	A = 0.
	B = 0.
	for(i in 1:size(data)[1])
		A += data[i, 2]*sin(2*pi*freq*data[i, 1])#/data[i, 3]^2
		B += data[i, 2]*cos(2*pi*freq*data[i, 1])#/data[i, 3]^2
	end
	return (A^2 + B^2)*4/size(data)[1]^2#/sum(1./data[:,3].^2)*4/size(data)[1]^2
end

# Compute the periodogram over the given frequency range
function periodogram(freq_min::Float64, freq_max::Float64,
							data::Matrix{Float64}, N::Int64=1000)
	# Frequency spacing
	df = (freq_max - freq_min)/(N - 1)

	freq = Array(Float64, (N, ))
	pgram = Array(Float64, (N, ))
	for(i in 1:N)
		freq[i] = freq_min + (i-1)*df
		pgram[i] = power(freq[i], data)
	end

	return (freq, pgram)
end


# Compute the "periodogram" over the given frequency range
function not_periodogram(freq_min::Float64, freq_max::Float64,
							data::Matrix{Float64},
							pgram_init::Vector{Float64}, N::Int64=1000)
	# Frequency spacing
	df = (freq_max - freq_min)/(N - 1)

	plt.ion()
	plt.hold(false)
	freq = Array(Float64, (N, ))
	pgram = Array(Float64, (N, 2))
	logl = Array(Float64, (N, ))
	for(i in 1:N)
		freq[i] = freq_min + (i-1)*df
		result = fit_mode(freq[i], data, sqrt(pgram_init[i]))
		pgram[i, :] = result[1:2]
		logl[i] = -result[3]
		plt.plot(freq[1:i], exp(logl[1:i] - maximum(logl[1:i])))
		plt.draw()
	end
	plt.ioff()
	plt.show()

	return (freq, pgram, logl)
end


###############################################
# MAIN PROGRAM
###############################################

using PyCall
@pyimport matplotlib.pyplot as plt

# Load the data and plot the periodogrm
data = readdlm("two_modes.txt")

nu_min = 0.5
nu_max = 1.5
N = 101

(freq, pgram_init) = periodogram(nu_min, nu_max, data, N)
plt.plot(freq, sqrt(pgram_init))
plt.show()

(freq, pgram, logl) = not_periodogram(nu_min, nu_max, data, pgram_init, N)

plt.plot(freq, exp(logl-max(logl))
plt.show()


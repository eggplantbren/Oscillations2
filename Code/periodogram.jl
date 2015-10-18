# Compute the 'power' at a given frequency
function power(freq::Float64, data::Matrix{Float64})
	A = 0.
	B = 0.
	for(i in 1:size(data)[1])
		A += data[i, 2]*sin(2*pi*freq*data[i, 1])/data[i, 3]^2
		B += data[i, 2]*cos(2*pi*freq*data[i, 1])/data[i, 3]^2
	end
	return (A^2 + B^2)/sum(1./data[:,3].^2)
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


###############################################
# MAIN PROGRAM
###############################################

using PyCall
@pyimport matplotlib.pyplot as plt

# Load the data and plot the periodogrm
data = readdlm("data.txt")

(freq, pgram) = periodogram(0.0, 0.2, data)

plt.plot(freq, pgram)
plt.xlabel("Frequency")
plt.show()


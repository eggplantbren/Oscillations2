from pylab import *

posterior_sample = atleast_2d(loadtxt('posterior_sample.txt'))
N_max = posterior_sample[0,1]

log_periods = array([])
amplitudesquared = array([])
mode_lifetimes = array([])

for i in xrange(0, posterior_sample.shape[0]):
  N = posterior_sample[i, 7]
  log_periods = hstack([log_periods, posterior_sample[i, 8:8+N]])
  amplitudesquared = hstack([amplitudesquared, posterior_sample[i, 8+N_max:8+N_max+N]**2])
  mode_lifetimes = hstack([mode_lifetimes, exp(posterior_sample[i, 8:8+N])*posterior_sample[i, 8+2*N_max:8+2*N_max+N]])

figure(figsize=(10, 10))

# Frequencies in microhz
frequencies = 1E6*exp(-log_periods)

# Trim extremes that would affect histograms too much
s = sort(frequencies)
median = s[0.5*len(s)]
iqr = s[0.75*len(s)] - s[0.25*len(s)]
keep = logical_and(frequencies > median - 5*iqr,
					frequencies < median + 5*iqr)
frequencies = frequencies[keep]
amplitudesquared = amplitudesquared[keep]

subplot(2,1,1)
hist(frequencies, 300)
xlabel(r'Frequency ($\mu$Hz)', fontsize=16)
ylabel('Relative Probability', fontsize=16)

subplot(2,1,2)
hist(frequencies, 300, weights=amplitudesquared)
xlabel(r'Frequency ($\mu$Hz)', fontsize=16)
ylabel('Relative Power', fontsize=16)
show()

#hist(mode_lifetimes/86400., 100)
#show()


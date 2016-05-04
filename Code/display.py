from pylab import *

posterior_sample = atleast_2d(loadtxt('posterior_sample.txt'))
frequencies = posterior_sample[:,5:105]
mode_lifetime = posterior_sample[:,-1]

hist(frequencies.flatten()[frequencies.flatten() != 0.0]/1E-6, 300)
xlabel('Frequency ($\\mu$Hz)', fontsize=16)
ylabel('Relative Probability', fontsize=16)
xlim([0, 200])
show()

hist(mode_lifetime/86400, 100)
xlabel("Mode Lifetime (days)")
show()


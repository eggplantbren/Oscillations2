from pylab import *

posterior_sample = atleast_2d(loadtxt('posterior_sample.txt'))
N_max = posterior_sample[:,1]

log_periods = array([])

for i in xrange(0, posterior_sample.shape[0]):
  N = posterior_sample[i, 6]
  log_periods = hstack([log_periods, posterior_sample[i, 6:6+N]])

hist(log_periods, 200)
xlabel(r'$\ln$(Period)', fontsize=20)
show()

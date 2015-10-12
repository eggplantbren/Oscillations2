"""
Code implementing the analytic solution given by
arxiv: 1102.0524
"""

# Angular frequency of oscillator
omega0 = 1.0

# Mode lifetime
tau = 5.0

# Cyclic frequency of the damped oscillator
omega = sqrt(omega0^2 - 1.0/(4.0*tau^2))

# The matrices M, I, and J
M = [0.0 -1.0; omega0^2 1.0/tau]
I = eye(2)
J = [1.0/(2.0*omega*tau) 1.0/omega; -omega0^2/omega -1.0/(2.0*omega*tau)]


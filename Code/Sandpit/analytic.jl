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

Dt = 1.0

# Covariance matrix for (delta_x, delta_v)
C = Array(Float64, (2, 2))
C[1, 1] = D/(4*omega^2*omega0^2*tau^3)*
				(4*omega^2*tau^2 + exp(-Dt/tau)*
				(cos(2*omega*Dt) - 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
C[1, 2] = D/(omega^2*tau^2)*exp(-Dt/tau)*sin(omega*Dt)^2
C[2, 1] = C[1, 2]
C[2, 2] = D/(4*omega^2*tau^3)*
				(4*omega^2*tau^2 + exp(-Dt/tau)*
				(cos(2*omega*Dt) + 2*omega*tau*sin(2*omega*Dt) - 4*omega0^2*tau^2))
println(C)


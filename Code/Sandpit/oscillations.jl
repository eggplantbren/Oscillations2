using PyCall
@pyimport matplotlib.pyplot as plt

function deriv(state::Array{Float64, 1}, dt)
	omega = (2*pi)/20.0
	tau = 30.0

	d = copy(state)
	d[1] = state[2]
	d[2] = -omega^2*state[1] - 1.0/tau*state[2] + randn()/sqrt(dt)
	return d
end

function update!(state::Array{Float64, 1}, dt)
	f1 = deriv(state, dt)
	f2 = deriv(state + 0.5*dt*f1, dt)
	f3 = deriv(state + 0.5*dt*f2, dt)
	f4 = deriv(state + dt*f3, dt)
	C = dt/6
	for(i in 1:length(state))
		state[i] += C*(f1[i] + 2*f2[i] + 2*f3[i] + f4[i])
	end
end


steps = 100000
skip = 100
plot_skip = 100
dt = 0.02

state = [0.0, 0.0]

keep = zeros((div(steps, skip), length(state)))

plt.ion()
plt.hold(false)
for(i in 1:steps)
	update!(state, dt)

	if(rem(i, skip) == 0)
		keep[div(i, skip), :] = state
		if(rem(i, skip*plot_skip) == 0)
			plt.plot((1:div(i, skip))*dt*skip, keep[1:div(i, skip), 1], "b")
			plt.draw()
		end
	end
end

data = zeros((size(keep)[1], 3))
data[:,1] = (1:size(keep)[1])*dt*skip
data[:,2] = keep[:,1] + 1E-3*randn(size(keep)[1])
data[:,3] = 1E-3

writedlm("data.txt", data)
plt.ioff()
plt.show()


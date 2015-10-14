using PyCall
@pyimport matplotlib.pyplot as plt

function deriv(state::Array{Float64, 1}, dt)
	d = copy(state)
	d[1] = state[2]
	d[2] = -state[1] - 0.1*state[2] + randn()/sqrt(dt)
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
skip = 10
dt = 0.05

state = [0.0, 0.0]

keep = zeros((div(steps, skip), length(state)))

plt.ion()
plt.hold(false)
for(i in 1:steps)
	update!(state, dt)

	if(rem(i, skip) == 0)
		keep[div(i, skip), :] = state
		plt.plot(keep[1:div(i, skip), 1], keep[1:div(i, skip), 2], "b")
		plt.draw()
		println(std(keep[1:div(i, skip)]))
	end
end


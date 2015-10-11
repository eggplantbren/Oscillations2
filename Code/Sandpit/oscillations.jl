using PyCall
@pyimport matplotlib.pyplot as plt

function deriv(state::Array{Float64, 1})
	d = copy(state)
	d[1] = state[2]
	d[2] = -state[1]
	return d
end

function update!(state::Array{Float64, 1}, dt=0.01)
	f1 = deriv(state)
	f2 = deriv(state + 0.5*dt*f1)
	f3 = deriv(state + 0.5*dt*f2)
	f4 = deriv(state + dt*f3)
	C = dt/6
	for(i in 1:length(state))
		state[i] += C*(f1[i] + 2*f2[i] + 2*f3[i] + f4[i])
	end
end


steps = 1000000
skip = 100
dt = 0.01

state = [1.0, 0.0]


keep = zeros((div(steps, skip), length(state)))

plt.ion()
plt.hold(false)
for(i in 1:steps)
	update!(state)

	if(rem(i, skip) == 0)
		keep[div(i, skip), :] = state
		plt.plot(keep[1:div(i, skip)], "b")
		plt.draw()
	end
end


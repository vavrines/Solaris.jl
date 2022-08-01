using Lux, OrdinaryDiffEq, Flux, Optimization, Optim, Random, Plots
using DiffEqFlux: NeuralODE
using Solaris: sci_train

rng = Random.default_rng()
Random.seed!(rng, 0)

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

nn = Lux.Chain(ActivationFunction(x -> x .^ 3), Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
p, st = Lux.setup(rng, nn)
p1 = Lux.ComponentArray(p)

#prob_neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
#predict_neuralode(p) = Array(prob_neuralode(u0, p, st)[1])
dudt(x, p, t) = nn(x, p, st) |> first
prob_node = ODEProblem(dudt, u0, tspan, p1)

function loss(p)
    pred = solve(prob_node, Midpoint(), u0 = u0, p = p, saveat = tsteps) |> Array
    #pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)

    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res =
    sci_train(loss, p, ADAM(0.05), Optimization.AutoZygote(); callback = cb, maxiters = 300)
res = sci_train(
    loss,
    res.u,
    LBFGS(),
    Optimization.AutoZygote();
    callback = cb,
    maxiters = 300,
)

#sol = prob_neuralode(u0, res.u, st)
sol = solve(prob_node, Midpoint(), u0 = u0, p = res.u, saveat = tsteps)
nde_data = sol |> Array

nde_data .- ode_data

plot(ode_data[1, :])
plot!(ode_data[2, :])
plot!(nde_data[1, :], line = :dash)
plot!(nde_data[2, :], line = :dash)

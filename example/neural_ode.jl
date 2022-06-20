using Lux, OrdinaryDiffEq, Flux, Optimization, Optim, Random
using DiffEqFlux: NeuralODE
using Solaris: sci_train

rng = Random.default_rng()

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

prob_neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)

predict_neuralode(p) = Array(prob_neuralode(u0, p, st)[1])

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = false)
    display(l)
    return false
end

sci_train(loss_neuralode, p, ADAM(0.05), Optimization.AutoZygote(); callback = callback, maxiters = 300)

res = sci_train(loss_neuralode, Lux.ComponentArray(p), ADAM(0.05), Optimization.AutoZygote(); callback = callback, maxiters = 300)
res = sci_train(loss_neuralode, res.u, LBFGS(), Optimization.AutoZygote(); callback = callback, maxiters = 300)

sol = prob_neuralode(u0, p, st)
sol[1]

sol = prob_neuralode(u0, res.u, st)
nde_data = sol[1] |> Array

nde_data .- ode_data

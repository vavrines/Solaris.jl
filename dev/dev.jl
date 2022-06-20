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

sci_train(loss_neuralode, p, ADAM(0.05), callback = callback, maxiters = 300)



# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p))

result_neuralode =
    Optimization.solve(optprob, ADAM(0.05), callback = callback, maxiters = 300)

optprob2 = remake(optprob, u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    LBFGS(),
    callback = callback,
    allow_f_increases = false,
)

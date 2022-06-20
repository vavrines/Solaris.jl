using Lux, OrdinaryDiffEq
using Flux

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(ActivationFunction(x -> x.^3),
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
                  p, st = Lux.setup(rng, dudt2)
                  
                  
                  prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

dudt2(randn(2), )

initial_params(dudt2)

Flux.destructure(dudt2)[1]

Lux.initialparameters(rng, dudt2)

using Random
rng = Random.default_rng()

p, st = Lux.setup(rng, dudt2)

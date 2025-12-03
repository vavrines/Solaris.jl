using Lux, Random, Solaris

init_params(Dense(21, 21))
init_params(Chain(Dense(21, 21), Dense(21, 21)))

nn = Chain(Dense(2, 4, tanh), Dense(4, 2))

rng = Random.default_rng()
Random.seed!(rng, 0)

Lux.setup(rng, nn)
Lux.initialparameters(rng, nn)
Lux.initialstates(rng, nn)

p, st = SR.setup(nn)
p1 = ComponentArray(p)

X = randn(Float32, 2, 10)
Y = rand(Float32, 2, 10)
sci_train(nn, (X, Y))

###
# self-defined layer
###
#--- definition ---#
using Zygote

struct LuxLinear <: Lux.AbstractLuxLayer
    init_A::Any
    init_B::Any
end

function LuxLinear(A::AbstractArray, B::AbstractArray)
    # Storing Arrays or any mutable structure inside a Lux Layer is not recommended
    # instead we will convert this to a function to perform lazy initialization
    return LuxLinear(() -> copy(A), () -> copy(B))
end

# `B` is a parameter
Lux.initialparameters(rng::AbstractRNG, layer::LuxLinear) = (B=layer.init_B(),)

# `A` is a state
Lux.initialstates(rng::AbstractRNG, layer::LuxLinear) = (A=layer.init_A(),)

(l::LuxLinear)(x, ps, st) = st.A * ps.B * x, st

#--- test ---#
model = LuxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
x = randn(rng, 2, 1)
ps, st = Lux.setup(rng, model)
model(x, ps, st)
gradient(ps -> sum(first(model(x, ps, st))), ps)

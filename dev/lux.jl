using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

x = Lux.Dense(2, 50)

Lux.setup(x) = Lux.setup(Random.default_rng(), x)

p, st = setup(x)
p, st = Lux.setup(rng, x)

Lux.initialparameters(rng, x)
Lux.initialstates(rng, x)

###
# self-defined layer
###
#--- definition ---#
using Zygote

struct LuxLinear <: Lux.AbstractExplicitLayer
    init_A::Any
    init_B::Any
end

function LuxLinear(A::AbstractArray, B::AbstractArray)
    # Storing Arrays or any mutable structure inside a Lux Layer is not recommended
    # instead we will convert this to a function to perform lazy initialization
    return LuxLinear(() -> copy(A), () -> copy(B))
end

# `B` is a parameter
Lux.initialparameters(rng::AbstractRNG, layer::LuxLinear) = (B = layer.init_B(),)

# `A` is a state
Lux.initialstates(rng::AbstractRNG, layer::LuxLinear) = (A = layer.init_A(),)

(l::LuxLinear)(x, ps, st) = st.A * ps.B * x, st

#--- test ---#
model = LuxLinear(randn(rng, 2, 4), randn(rng, 4, 2))
x = randn(rng, 2, 1)

ps, st = Lux.setup(rng, model)

model(x, ps, st)

gradient(ps -> sum(first(model(x, ps, st))), ps)

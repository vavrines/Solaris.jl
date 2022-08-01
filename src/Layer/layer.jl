abstract type AbstractLayer end
abstract type AbstractChain end
abstract type AbstractExplicitLayer end
abstract type AbstractExplicitChain end

include("dense.jl")
include("chain.jl")
include("flux.jl")
include("resnet.jl")
include("convex.jl")

init_params(c::AbstractExplicitChain) = vcat(init_params.(c.layers)...)
param_length(c::AbstractExplicitChain) = sum(param_length(x) for x in c.layers)

"""
The implementation is lengthy since broadcasting over dictionary and NamedTuple is reserved in Julia.
"""
function init_params(c::Lux.Chain)
    ps = init_params(first(c.layers))
    for i in 2:length(c.layers)
        ps = [ps; init_params(c.layers[i])]
    end

    return ps
end

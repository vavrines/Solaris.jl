abstract type AbstractLayer end
abstract type AbstractChain end
abstract type AbstractExplicitLayer end
abstract type AbstractExplicitChain end

include("flux.jl")
include("resnet.jl")
include("convex.jl")

init_params(c::AbstractExplicitChain) = vcat(init_params.(c.layers)...)
param_length(c::AbstractExplicitChain) = sum(param_length(x) for x in c.layers)

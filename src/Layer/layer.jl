abstract type AbstractLayer end
abstract type AbstractChain end

include("flux.jl")
include("affine.jl")
include("resnet.jl")
include("convex.jl")

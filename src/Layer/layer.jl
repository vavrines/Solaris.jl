abstract type AbstractNetwork end
abstract type AbstractLayer <: AbstractNetwork end
abstract type AbstractChain <: AbstractNetwork end
abstract type AbstractExplicitLayer <: AbstractLayer end
abstract type AbstractExplicitChain <: AbstractChain end

include("dense.jl")
include("chain.jl")
include("flux.jl")
include("lux.jl")
include("resnet.jl")
include("convex.jl")
include("pointnet.jl")

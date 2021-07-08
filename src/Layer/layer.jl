abstract type AbstractLayer end
abstract type AbstractChain end

# ------------------------------------------------------------
# Extended Flux.Chain
# ------------------------------------------------------------
function Flux.Chain(D::Integer, N::Integer, σ::Function)
    t = ()
    for i = 1:N
        t = (t..., Dense(D, D, σ))
    end

    return Chain(t...)
end

include("affine.jl")
include("resnet.jl")
include("convex.jl")

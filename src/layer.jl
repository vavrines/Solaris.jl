"""
    struct Affine{T1<:AbstractArray,T2}
        w::T1
        b::T2
        σ::Function
    end

Affine layer of neural network

"""
struct Affine{T1<:AbstractArray,T2}
    w::T1
    b::T2
    σ::Function

    function Affine(W::AbstractArray, B, S)
        new{typeof(W),typeof(B)}(W, B, S)
    end
end

function Affine(
    in::Integer,
    out::Integer,
    σ = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    isBias = true::Bool,
    precision = Float32,
)
    if isBias
        return Affine(fw(precision, out, in), fb(precision, out), σ)
    else
        return Affine(fw(precision, out, in), Flux.Zeros(out), σ)
    end
end

# ------------------------------------------------------------
# Functor
# ------------------------------------------------------------
Flux.@functor Affine # works with Flux.params

(L::Affine)(x::AbstractVector) = L.w * x .+ L.b

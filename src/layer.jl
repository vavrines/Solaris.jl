"""
Affine layer of neural network

@args: W, b, σ

"""
struct Affine{T1,T2,T3<:Function}
    W::T1
    b::T2
    σ::T3
end

Affine(W, b) = Affine(W, b, identity) # @func identity returns its args

function Affine(
    in::Integer,
    out::Integer,
    σ = identity::Function;
    funcW = randn::Function,
    funcB = randn::Function,
    isBias = true::Bool,
)
    if isBias
        return Affine(funcW(out, in), funcB(out), σ)
    else
        return Affine(funcW(out, in), Flux.Zeros(out), σ)
    end
end


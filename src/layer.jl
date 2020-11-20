"""
Affine layer of neural network

@args: w, b, σ

"""
struct Affine{T1,T2}
    w::T1
    b::T2
    σ::Function

    function Affine(
        W::AbstractArray,
        B::AbstractArray,
        FUNC=identity::Function,
    )
        return new{typeof(W),typeof(B)}(W, B, FUNC)
    end

    function Affine(
        in::Integer,
        out::Integer,
        σ = identity::Function;
        fw = randn::Function,
        fb = randn::Function,
        isBias = true::Bool,
    )
        if isBias
            return Affine(fw(out, in), fb(out), σ)
        else
            return Affine(fw(out, in), Flux.Zeros(out), σ)
        end
    end
end

(L::Affine)(x) = L.w * x .+ L.b
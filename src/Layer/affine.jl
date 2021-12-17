"""
    struct Affine
        w
        b
        σ
    end

Affine layer of neural network

"""
struct Affine{T1<:AbstractArray,T2}
    w::T1
    b::T2
    σ::Function
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
        return Affine(fw(precision, out, in), Flux.Zeros(precision, out), σ)
    end
end

Flux.@functor Affine # works with Flux.params

(L::Affine)(x::AbstractVector) = L.w * x .+ L.b


"""
    struct FastAffine{I,F,F2} <: DiffEqFlux.FastLayer
        out::I
        in::I
        σ::F
        initial_params::F2
    end
Equivalent FastDense layer with controllable type
"""
struct FastAffine{I,F,F2} <: DiffEqFlux.FastLayer
    out::I
    in::I
    σ::F
    initial_params::F2
end

function FastAffine(
    in::Integer,
    out::Integer,
    σ = identity;
    fw = randn,
    fb = Flux.zeros,
    precision = Float32,
)
    initial_params() = vcat(vec(fw(precision, out, in)), fb(precision, out))
    return FastAffine(out, in, σ, initial_params)
end

(f::FastAffine)(x, p) =
    f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])

DiffEqFlux.paramlength(f::FastAffine) = f.out * (f.in + 1)
DiffEqFlux.initial_params(f::FastAffine) = f.initial_params()

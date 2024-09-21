"""
$(TYPEDEF)

Functional dense layer

## Fields

- `in`: input size
- `out`: output size
- `σ`: activation function
- `initial_params`: initialization method
- `bias`: whether to include a bias term
"""
struct FnDense{F,F2} <: AbstractExplicitLayer
    in::Int
    out::Int
    σ::F
    initial_params::F2
    bias::Bool

    function FnDense(
        in::Integer,
        out::Integer,
        σ = identity;
        bias = true,
        initW = Flux.glorot_uniform,
        initb = Flux.zeros32,
    )
        temp = (
            (bias == false) ? vcat(vec(initW(out, in))) :
            vcat(vec(initW(out, in)), initb(out))
        )
        initial_params() = temp

        _σ = fast_act(σ)
        new{typeof(_σ),typeof(initial_params)}(in, out, _σ, initial_params, bias)
    end
end

(f::FnDense)(x::Number, p) = (
    (f.bias == true) ?
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x))
)

(f::FnDense)(x::AbstractVector, p) = (
    (f.bias == true) ?
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x))
)

(f::FnDense)(x::AbstractMatrix, p) = (
    (f.bias == true) ?
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
    (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x))
)

param_length(f::FnDense) = f.out * (f.in + f.bias)

init_params(f::FnDense) = f.initial_params()

apply(m::FnDense, x, p) = m(x, p)

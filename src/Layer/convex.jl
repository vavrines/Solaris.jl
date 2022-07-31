"""
$(TYPEDEF)
    
Input Convex Neural Network (ICNN) layer by Amos et al.
"""
struct Convex{T1<:AbstractArray,T2<:Union{Flux.Zeros,AbstractVector},T3} <: AbstractLayer
    W::T1
    U::T1
    b::T2
    σ::T3
end

function Convex(
    z_in::T,
    x_in::T,
    out::T,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {T<:Integer}
    return Convex(
        fw(precision, out, z_in),
        fw(precision, out, x_in),
        fb(precision, out),
        activation,
    )
end

function (m::Convex)(x::AbstractArray)
    W, b, σ = m.W, m.b, m.σ
    sz = size(x)
    x = reshape(x, sz[1], :) # reshape to handle dims > 1 as batch dimensions 
    x = σ.(W * x .+ b)

    return reshape(x, :, sz[2:end]...)
end

function (m::Convex)(z::AbstractArray, x::AbstractArray)
    W, U, b, σ = m.W, m.U, m.b, m.σ
    sz = size(z)
    sx = size(x)
    z = reshape(z, sz[1], :)
    x = reshape(x, sx[1], :)
    z = σ.(softplus.(W) * z + U * x .+ b)

    return reshape(z, :, sz[2:end]...)
end

Flux.@functor Convex

function Base.show(io::IO, model::Convex{T1,T2,T3}) where {T1,T2,T3}
    print(
        io,
        "Input convex layer{$T1,$T2,$T3}\n",
        "nonnegative weights for: $(model.W |> size)\n",
        "input weights: $(model.U |> size)\n",
        "bias: $(model.b |> size)\n",
        "activation: $(model.σ)\n",
    )
end

struct ICNN{T} <: AbstractChain
    layers::T
end

function ICNN(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}

    layers = (Dense(din, layer_sizes[1], activation),)

    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            layers = (
                layers...,
                Convex(
                    layer_sizes[i],
                    din,
                    out,
                    activation;
                    fw = fw,
                    fb = fb,
                    precision = precision,
                ),
            )
            i += 1
        end
    end
    layers = (
        layers...,
        Convex(
            layer_sizes[end],
            din,
            dout,
            identity;
            fw = fw,
            fb = fb,
            precision = precision,
        ),
    )

    return ICNN(layers)

end

(m::ICNN)(x::AbstractArray) = begin
    z = m.layers[1](x)
    for i = 2:length(m.layers)
        z = m.layers[i](z, x)
    end
    return z
end

Flux.@functor ICNN

function Base.show(io::IO, model::ICNN{T}) where {T}
    print(
        io,
        "Input convex neural network{$T}\n",
        "Layers: 1 Dense layer + $(length(model.layers)-1) convex layers\n",
    )
end


"""
$(TYPEDEF)
    
Fast ICNN layer
"""
struct FastConvex{I<:Integer,F1,F2} <: AbstractExplicitLayer
    zin::I
    xin::I
    out::I
    σ::F1
    initial_params::F2
end

function FastConvex(
    zin::Integer,
    xin::Integer,
    out::Integer,
    σ = identity;
    fw = randn,
    fb = zeros,
    precision = Float32,
)
    initial_params() =
        vcat(vec(fw(precision, out, zin)), vec(fw(precision, out, xin)), fb(precision, out))
    return FastConvex{typeof(out),typeof(σ),typeof(initial_params)}(
        zin,
        xin,
        out,
        σ,
        initial_params,
    )
end

function (f::FastConvex)(z, x, p)
    f.σ.(
        softplus.(reshape(p[1:(f.out*f.zin)], f.out, f.zin)) * z .+
        reshape(p[f.out*f.zin+1:(f.out*f.zin+f.out*f.xin)], f.out, f.xin) * x .+
        p[(f.out*f.zin+f.out*f.xin+1):end],
    )
end

DiffEqFlux.paramlength(f::FastConvex) = f.out * (f.zin + f.xin + 1)
DiffEqFlux.initial_params(f::FastConvex) = f.initial_params()
param_length(f::FastConvex) = f.out * (f.zin + f.xin + 1)
param_length(f::DiffEqFlux.FastDense) = f.out*(f.in+f.bias)
init_params(f::FastConvex) = f.initial_params()
init_params(f::DiffEqFlux.FastDense) = f.initial_params()

struct FastICNN{T} <: AbstractExplicitChain
    layers::T
end

function FastICNN(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}

    layers = (FastDense(din, layer_sizes[1], activation),)

    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            layers = (
                layers...,
                FastConvex(
                    layer_sizes[i],
                    din,
                    out,
                    activation;
                    fw = fw,
                    fb = fb,
                    precision = precision,
                ),
            )
            i += 1
        end
    end
    layers = (
        layers...,
        FastConvex(
            layer_sizes[end],
            din,
            dout,
            identity;
            fw = fw,
            fb = fb,
            precision = precision,
        ),
    )

    return FastICNN(layers)

end

DiffEqFlux.initial_params(c::FastICNN) = vcat(initial_params.(c.layers)...)

#=function (m::FastICNN)(x::AbstractArray, p)
    z = m.layers[1](x, p[1:DiffEqFlux.paramlength(m.layers[1])])
    counter = DiffEqFlux.paramlength(m.layers[1])
    for i = 2:length(m.layers)
        z = m.layers[i](z, x, p[counter+1:counter+DiffEqFlux.paramlength(m.layers[i])])
        counter += DiffEqFlux.paramlength(m.layers[i])
    end

    return z
end=#

function (m::FastICNN)(x::AbstractArray, p)
    z = m.layers[1](x, p[1:param_length(m.layers[1])])
    counter = param_length(m.layers[1])
    for i = 2:length(m.layers)
        z = m.layers[i](z, x, p[counter+1:counter+param_length(m.layers[i])])
        counter += param_length(m.layers[i])
    end

    return z
end

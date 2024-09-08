using Lux, Random#, Solaris

# Dense

rng = Random.default_rng()
Random.seed!(rng, 0)

d = Lux.Dense(4, 4, tanh)
Lux.setup(rng, d)

dd = Lux.Chain(d, d)
Lux.setup(rng, dd)

# Convex

struct FConvex{I<:Integer,F1,F2} <: Lux.AbstractLuxLayer
    zin::I
    xin::I
    out::I
    σ::F1
    initial_params::F2
end

function FConvex(
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
    return FConvex{typeof(out),typeof(σ),typeof(initial_params)}(
        zin,
        xin,
        out,
        σ,
        initial_params,
    )
end

function (f::FConvex)(z, x, p)
    f.σ.(
        softplus.(reshape(p[1:(f.out*f.zin)], f.out, f.zin)) * z .+
        reshape(p[f.out*f.zin+1:(f.out*f.zin+f.out*f.xin)], f.out, f.xin) * x .+
        p[(f.out*f.zin+f.out*f.xin+1):end],
    )
end

Lux.initialparameters(rng::AbstractRNG, m::FConvex) = (p = m.initial_params(),)
Lux.initialstates(rng, x) = NamedTuple()

m = FConvex(4, 4, 4)
Lux.setup(rng, m)

# ICNN

struct FICNN{T} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function FICNN(
    din::TI,
    dout::TI,
    layer_sizes::TT,
    activation = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    precision = Float32,
) where {TI<:Integer,TT<:Union{Tuple,AbstractVector}}

    layers = (Lux.Dense(din, layer_sizes[1], activation),)

    if length(layer_sizes) > 1
        i = 1
        for out in layer_sizes[2:end]
            layers = (
                layers...,
                FConvex(
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
        FConvex(
            layer_sizes[end],
            din,
            dout,
            identity;
            fw = fw,
            fb = fb,
            precision = precision,
        ),
    )

    return FICNN(layers)

end

nn = FICNN(4, 4, [10, 10, 10])

Lux.setup(rng, nn)

Lux.initialparameters(rng, nn)

Lux.initialparameters(rng, dd)

p = Solaris.init_params(nn)
X = randn(Float32, 4, 10)
Y = rand(Float32, 4, 10)

nn(X, p)

sci_train(nn, (X, Y))

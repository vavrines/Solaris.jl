getfn(x) = x
getfn(x::Tuple) = first(x)

apply(::Tuple{}, x, p) = x
apply(fs::Tuple, x, p) = apply(
    Base.tail(fs),
    first(fs)(x, p[1:param_length(first(fs))]),
    p[(param_length(first(fs))+1):end],
)

struct FnChain{T<:Tuple} <: AbstractExplicitChain
    layers::T

    function FnChain(xs...)
        layers = getfn.(xs)
        new{typeof(layers)}(layers)
    end
end

(c::FnChain)(x, p) = apply(c.layers, x, p)

param_length(c::FnChain) = sum(param_length(x) for x in c.layers)

init_params(c::FnChain) = vcat(init_params.(c.layers)...)

apply(c::Lux.Chain, x, p, st) = c(x, p, st)[1]

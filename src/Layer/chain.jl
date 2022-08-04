getfunc(x) = x
getfunc(x::Tuple) = first(x)

applychain(::Tuple{}, x, p) = x
applychain(fs::Tuple, x, p) = applychain(
    Base.tail(fs),
    first(fs)(x, p[1:param_length(first(fs))]),
    p[(param_length(first(fs))+1):end],
)

struct FnChain{T<:Tuple} <: AbstractExplicitChain
    layers::T

    function FnChain(xs...)
        layers = getfunc.(xs)
        new{typeof(layers)}(layers)
    end
end

(c::FnChain)(x, p) = applychain(c.layers, x, p)

param_length(c::FnChain) = sum(param_length(x) for x in c.layers)

init_params(c::FnChain) = vcat(init_params.(c.layers)...)

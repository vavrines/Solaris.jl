struct FastChain{T<:Tuple}
    layers::T
    function FastChain(xs...)
        layers = getfunc.(xs)
        new{typeof(layers)}(layers)
    end
end
getfunc(x) = x
getfunc(x::Tuple) = first(x)

applychain(::Tuple{}, x, p) = x
applychain(fs::Tuple, x, p) = applychain(
    Base.tail(fs),
    first(fs)(x, p[1:param_length(first(fs))]),
    p[(param_length(first(fs))+1):end],
)
(c::FastChain)(x, p) = applychain(c.layers, x, p)
param_length(c::FastChain) = sum(paramlength(x) for x in c.layers)
init_params(c::FastChain) = vcat(init_params.(c.layers)...)

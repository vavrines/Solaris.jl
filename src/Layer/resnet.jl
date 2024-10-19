"""
$(TYPEDEF)

Shortcut connection for ResNet-type blocks
"""
struct Shortcut{T}
    chain::T
    f::Function
    σ::Function
end

Shortcut(chain::T) where {T} = Shortcut{typeof(chain)}(chain, +, tanh)

Flux.@functor Shortcut

(nn::Shortcut)(x) = nn.σ.(nn.f(nn.chain(x), x))

function Base.show(io::IO, model::Shortcut{T}) where {T}
    return print(
        io,
        "Shortcut{$T}\n",
        "chain: $(model.chain)\n",
        "connection: $(model.f)\n",
        "activation: $(model.σ)\n",
    )
end

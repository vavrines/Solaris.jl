"""
Extended Flux.Chain method
"""
function Flux.Chain(D::Integer, N::Integer, σ::Function)
    t = ()
    for i = 1:N
        t = (t..., Flux.Dense(D, D, σ))
    end

    return Flux.Chain(t...)
end


"""
$(SIGNATURES)

Create dense layer with meticulous settings
"""
function dense_layer(
    in::T,
    out::T,
    σ = identity::Function;
    fw = randn::Function,
    fb = zeros::Function,
    isBias = true::Bool,
    precision = Float32,
) where {T<:Integer}
    if isBias
        return Flux.Dense(fw(precision, out, in), fb(precision, out), σ)
    else
        return Flux.Dense(fw(precision, out, in), Flux.Zeros(precision, out), σ)
    end
end

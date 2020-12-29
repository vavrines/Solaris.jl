"""
Illustrative examples from Flux.jl
"""

using Solaris
using Solaris.Flux, Solaris.DiffEqFlux

"""
gradient: scalar -> scalar
"""
gradient(x -> 3x^2 + 2x + 1, 5) # (32, )
gradient((a, b) -> a*b, 2, 3) # (3, 2)

"""
gradient: scalar -> vector

    f = (x₁ - y₁)² + (x₂ - y₂)²
    ∂f/∂x₁ = 2x₁ - 2y₁
    ⋯
"""
f(x, y) = sum((x .- y).^2)
gradient(f, [2, 1], [2, 0])

"""
params: a syntax sugar for machine learning with thousands of parameters
"""
x = [2, 1]
y = [2, 0]
gs = gradient(params(x, y)) do
    f(x, y)
end
gs[x]
gs[y]

"""
toy model
"""
W = rand(2, 5)
b = rand(2)

predict(x) = W * x .+ b
function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2)
end

X, Y = rand(5), rand(2)
L0 = loss(X, Y)

gs = gradient(() -> loss(X, Y), params(W, b))
W̄ = gs[W]
W .-= 0.1 .* W̄
L1 = loss(X, Y)

@assert L1 < L0
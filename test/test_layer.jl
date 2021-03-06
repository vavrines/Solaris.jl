using Flux
using Flux.Losses: logitcrossentropy
using DiffEqFlux

m = Chain(
    Solaris.Affine(28^2, 128, relu),
    Solaris.Affine(128, 32, relu),
    Solaris.Affine(32, 10),
)

m(randn(784))

# regularization
sqnorm(x) = sum(abs2, x)
loss(x, y) = logitcrossentropy(m(x), y) + sum(sqnorm, Flux.params(m))

loss(rand(Float32, 28^2), rand(Float32, 10))
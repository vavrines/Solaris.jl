using Solaris
using Solaris.Flux
using Solaris.Flux.Losses: logitcrossentropy
using Solaris.DiffEqFlux
using Test

m = Chain(
    Affine(28^2, 128, relu),
    Affine(128, 32, relu),
    Affine(32, 10),
)

# regularization
sqnorm(x) = sum(abs2, x)
loss(x, y) = logitcrossentropy(m(x), y) + sum(sqnorm, Flux.params(m))

loss(rand(Float32, 28^2), rand(Float32, 10))
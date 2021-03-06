using Flux
using Flux.Losses: logitcrossentropy
using DiffEqFlux

m = Chain(
    Solaris.Affine(28^2, 128, relu),
    Solaris.Affine(128, 32, relu),
    Solaris.Affine(32, 10),
)

m(randn(Float32, 784))

# regularization
sqnorm(x) = sum(abs2, x)
loss(x, y) = logitcrossentropy(m(x), y) + sum(sqnorm, Flux.params(m))

loss(rand(Float32, 28^2), rand(Float32, 10))

X = rand(Float32, 4, 100)
Y = X.^2
data = Flux.Data.DataLoader(X, Y, batchsize = 4, shuffle = true)

m = Solaris.ICNN(4, 4, [10, 10, 10, 10], tanh)
ps = params(m)
loss(x, y) = sum(abs2, m(x) .- y) / 100
cb = () -> println("loss: $(loss(X, Y))")

Flux.@epochs 1 Flux.train!(loss, ps, data, ADAM(), cb = Flux.throttle(cb, 1))
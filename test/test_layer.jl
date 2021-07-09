using Flux.Losses: logitcrossentropy
using DiffEqFlux

# affine
Affine(2, 1; isBias = true)
m = Affine(2, 1; isBias = false)
m(randn(Float32, 2))

# icnn
X = rand(Float32, 4, 10)
Y = X .^ 2
data = Flux.Data.DataLoader((X, Y), batchsize = 4, shuffle = true)
m = Solaris.ICNN(4, 4, [10, 10, 10, 10], tanh)

# fast icnn
fm = Solaris.FastICNN(4, 4, [10, 10, 10, 10], tanh)

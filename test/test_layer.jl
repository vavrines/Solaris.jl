using Flux.Losses: logitcrossentropy

# affine
Solaris.Affine(2, 1; isBias = true)
m = Solaris.Affine(2, 1; isBias = false)
m(randn(Float32, 2))

Chain(4, 4, tanh)
Solaris.dense_layer(4, 4; isBias = true)
Solaris.dense_layer(4, 4; isBias = false)

faf = Solaris.FastAffine(4, 4, tanh)
DiffEqFlux.paramlength(faf)
_p = initial_params(faf)
faf(randn(Float32, 4), _p)

nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
sm = Shortcut(nn)
show(sm)
sm(rand(21))

# icnn
icnnl = Convex(4, 4, 1, identity; fw = randn, fb = zeros, precision = Float32)
icnnc = ICNN(4, 1, [10, 10], identity; fw = randn, fb = zeros, precision = Float32)
show(icnnl)
show(icnnc)
icnnl(randn(4))
icnnl(randn(4), randn(4))
icnnc(randn(4))

# fast icnn
fil = FastConvex(4, 4, 4)
fic = FastICNN(4, 1, [10, 10])
fil(rand(4), rand(4), initial_params(fil))
fic(rand(4), initial_params(fic))

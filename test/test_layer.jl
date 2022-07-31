using Flux.Losses: logitcrossentropy

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

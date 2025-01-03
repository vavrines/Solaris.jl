using Flux.Losses: logitcrossentropy

# icnn
icnnl = Convex(4, 4, 1, identity; fw=randn, fb=zeros, precision=Float32)
icnnc = ICNN(4, 1, [10, 10], identity; fw=randn, fb=zeros, precision=Float32)
show(icnnl)
show(icnnc)
icnnl(randn(4))
icnnl(randn(4), randn(4))
icnnc(randn(4))

# fast icnn
fil = FastConvex(4, 4, 4)
fic = FastICNN(4, 1, [10, 10])
fil(rand(4), rand(4), init_params(fil))
fic(rand(4), init_params(fic))

# Lux
m = Lux.Dense(2, 2)
SR.init_params(m)
ps, st = SR.setup(m)
SR.stateful(m, st)
SR.stateful(m, ps, st)

# Flux
SR.dense_layer(2, 2)
m = Flux.Chain(2, 2, tanh)

# Resnet
Shortcut(m, +, tanh)

# PointNet
m = SR.PointNet()
X = rand(Float32, 3, 10, 2)
Y = rand(Float32, 3, 20, 1)
m(X)
m(Y)

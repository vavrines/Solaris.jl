nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
nn1 = FastChain(FastDense(21, 21, tanh), FastDense(21, 21))

X = randn(Float32, 21, 10)
Y = rand(Float32, 21, 10)
sci_train!(nn, (X, Y), ADAM())
sci_train!(nn, Flux.Data.DataLoader((X, Y)), ADAM(); device = cpu, epoch = 1)
sci_train(nn1, (X, Y))

loss(p) = nn1(X, p) |> sum
p1 = initial_params(nn1)
sci_train(loss, p1)

cd(@__DIR__)
model = load_model("model.h5"; mode = :tf)
sci_train!(model, (randn(Float32, 1, 4), randn(Float32, 1, 1)))

cb = function(args...)
    plot(randn(2))
    return false
end
vis_train(loss, p1, ADAM(); cb = cb, maxiters = 1)

nn = Chain(Dense(21, 21, tanh), Dense(21, 21))
nn1 = FnChain(FnDense(21, 21, tanh), FnDense(21, 21))

X = randn(Float32, 21, 10)
Y = rand(Float32, 21, 10)
sci_train!(nn, (X, Y), Adam())
sci_train!(nn, Flux.Data.DataLoader((X, Y)), Adam(); device = cpu, epoch = 1)
sci_train(nn1, (X, Y))

loss(p) = nn1(X, p) |> sum
p1 = init_params(nn1)
sci_train(loss, p1)

loss1(p, x, y) = sum(abs2, nn1(x, p) - y)
sci_train(loss1, p1, (X, Y), Adam())

cd(@__DIR__)
model = load_model("model.h5"; mode = :tf)
sci_train!(model, (randn(Float32, 1, 4), randn(Float32, 1, 1)))

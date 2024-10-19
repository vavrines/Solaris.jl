using Solaris, CUDA
using Solaris.Flux: DataLoader, cpu, gpu

nn = FnChain(FnDense(10, 20, tanh), FnDense(20, 10))
p0 = init_params(nn)

#--- full ---#
X = randn(Float32, 10, 1000000)
Y = rand(Float32, 10, 1000000)

function loss(p)
    pred = nn(X, p)
    return sum(abs2, pred - Y)
end

cb = function (p, l)
    display(l)
    return l < 1e-4
end

@time sci_train(loss, p0, Adam(); iters=10, cb=cb) # ~4.9s

function loss1(p)
    pred = nn(gpu(X), p)
    return sum(abs2, pred - gpu(Y))
end

@time sci_train(loss1, p0, Adam(); iters=10, cb=cb, device=gpu) # ~0.34s

#--- batch ---#
train_loader = DataLoader((X, Y); batchsize=100)

function loss1(p, data)
    batch_x, batch_y = data
    pred = nn(batch_x, p)
    return sum(abs2, pred - batch_y)
end

@time sci_train(loss1, p0, train_loader, Adam(); iters=10, cb=cb) # ~0.04s
@time sci_train(loss1, p0, train_loader, Adam(); iters=10, cb=cb, epochs=2) # there is a bug
@time sci_train(loss1, p0, (X, Y), Adam(); iters=10, cb=cb, batch=100)
@time sci_train(loss1, p0, train_loader, Adam(); device=gpu, iters=10, cb=cb, batch=100)

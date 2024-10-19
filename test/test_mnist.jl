using Solaris
using Flux, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using MLDatasets

mutable struct Args
    η::Float64      # learning rate
    batchsize::Int   # batch size
    epochs::Int        # number of epochs
    device::Function
end

function prepare_data(args)
    MLDatasets.MNIST.download(; i_accept_the_terms_of_use=true)

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader(xtrain, ytrain; batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest; batchsize=args.batchsize)

    return train_data, test_data
end

function build_model(; imgsize=(28, 28, 1), nclasses=10)
    return Chain(Affine(prod(imgsize), 32, relu), Affine(32, nclasses))
end

function loss_all(dataloader, model)
    l = 0.0f0
    for (x, y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    return l / length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x, y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) * 1 / size(x, 2)
    end
    return acc / length(data_loader)
end

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
# initialize parameters 
args = Args(1e-3, 1024, 10, cpu)

# load Data
train_data, test_data = prepare_data(args)
train_data = args.device.(train_data)
test_data = args.device.(test_data)

# construct model
m = build_model()
m = args.device(m)

# optimization
loss(x, y) = logitcrossentropy(m(x), y)
evalcb = () -> @show(loss_all(train_data, m))
opt = Adam()

# training
@epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb=evalcb)

@show accuracy(train_data, m)
@show accuracy(test_data, m)

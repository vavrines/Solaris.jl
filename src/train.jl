# ============================================================
# Training & Optimization Methods
# ============================================================

"""
$(SIGNATURES)

Scientific machine learning training function

- @arg ann: neural network model
- @arg data: tuple (X, Y) of dataset
- @arg θ: parameters of neural network
- @arg opt: optimizer
- @arg adtype: automatical differentiation type
- @arg args: rest arguments
- @arg device: cpu / gpu
- @arg maxiters: maximal iteration number
- @arg kwargs: keyword arguments
"""
function sci_train(
    ann::T,
    data,
    θ = initial_params(ann),
    opt = ADAM(),
    adtype = GalacticOptim.AutoZygote(),
    args...;
    device = cpu,
    maxiters = 200::Integer,
    kwargs...,
) where {T<:DiffEqFlux.FastLayer}
    data = data |> device
    θ = θ |> device
    L = size(data[1], 2)
    loss(p) = sum(abs2, ann(data[1], p) - data[2]) / L

    cb = function (p, l)
        println("loss: $(loss(p))")
        return false
    end

    f = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    fi = GalacticOptim.instantiate_function(f, θ, adtype, nothing)
    prob = GalacticOptim.OptimizationProblem(fi, θ; kwargs...)

    return GalacticOptim.solve(prob, opt, args...; cb = Flux.throttle(cb, 1), maxiters = maxiters, kwargs...)
end

"""
$(SIGNATURES)
"""
function sci_train(loss::Function, θ, opt = ADAM(), adtype = GalacticOptim.AutoZygote(), args...; maxiters = 200::Integer, kwargs...)
    f = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    fi = GalacticOptim.instantiate_function(f, θ, adtype, nothing)
    prob = GalacticOptim.OptimizationProblem(fi, θ; kwargs...)
    return GalacticOptim.solve(prob, opt, args...; maxiters = maxiters, kwargs...)
end


"""
$(SIGNATURES)

Scientific machine learning training function

- @arg ann: neural network model
- @arg data: tuple (X, Y) of dataset
- @arg opt: optimizer 
- @arg epoch: epoch number
- @arg batch: batch size
- @arg device: cpu / gpu
"""
function sci_train!(ann, data::Tuple, opt = ADAM(); device = cpu, epoch = 1, batch = 1)
    X, Y = data |> device
    L = size(X, 2)
    data = Flux.Data.DataLoader((X, Y), batchsize = batch, shuffle = true) |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs epoch Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

"""
$(SIGNATURES)
"""
function sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(); device = cpu, epoch = 1)
    X, Y = dl.data |> device
    L = size(X, 2)
    dl = dl |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    Flux.@epochs epoch Flux.train!(loss, ps, dl, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

"""
$(SIGNATURES)

Trainer for Tensorflow model
"""
function sci_train!(ann::PyObject, data::Tuple; device = cpu, split = 0.0, epoch = 1, batch = 64, verbose = 1)
    X, Y = data
    ann.fit(X, Y, validation_split=split, epochs=epoch, batch_size=batch, verbose=verbose)

    return nothing
end

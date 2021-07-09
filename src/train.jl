# ============================================================
# Training & Optimization Methods
# ============================================================

"""
    sci_train(
        ann::T,
        data,
        θ = initial_params(ann),
        opt = ADAM(),
        adtype = GalacticOptim.AutoZygote(),
        args...;
        maxiters = 200::Integer,
        kwargs...,
    ) where {T<:DiffEqFlux.FastLayer}

    sci_train(loss::Function, θ, opt, adtype = GalacticOptim.AutoZygote(), args...; kwargs...)

Scientific machine learning training function

- @args ann: neural network model
- @args data: tuple (X, Y) of dataset
- @args θ: parameters of neural network
- @args opt: optimizer
- @args adtype: automatical differentiation type
- @args args: rest arguments
- @args device: cpu / gpu
- @args maxiters: maximal iteration number
- @args kwargs: keyword arguments

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

function sci_train(loss::Function, θ, opt = ADAM(), adtype = GalacticOptim.AutoZygote(), args...; maxiters = 200::Integer, kwargs...)
    f = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    fi = GalacticOptim.instantiate_function(f, θ, adtype, nothing)
    prob = GalacticOptim.OptimizationProblem(fi, θ; kwargs...)
    return GalacticOptim.solve(prob, opt, args...; maxiters = maxiters, kwargs...)
end


"""
    sci_train!(ann, data::Tuple, opt = ADAM(); device = cpu, epoch = 1, batch = 1)
    sci_train!(ann, dl::Flux.Data.DataLoader, opt = ADAM(); device = cpu, epoch = 1)

Scientific machine learning training function

- @args ann: neural network model
- @args data: tuple (X, Y) of dataset
- @args opt: optimizer 
- @args epoch: epoch number
- @args batch: batch size
- @args device: cpu / gpu

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

# ------------------------------------------------------------
# TensorFlow
# ------------------------------------------------------------
function sci_train!(ann::PyObject, data::Tuple; device = cpu, split = 0.0, epoch = 1, batch = 64, verbose = 1)
    X, Y = data
    ann.fit(X, Y, validation_split=split, epochs=epoch, batch_size=batch, verbose=verbose)

    return nothing
end

# ------------------------------------------------------------
# Visualization training
# ------------------------------------------------------------
struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))

"""
Interactive training with solution animation

"""
function vis_train(
    loss,
    _θ,
    opt,
    _data = DEFAULT_DATA;
    cb = (args...) -> false,
    maxiters = get_maxiters(data),
    progress = true,
    save_best = true,
)
    θ = copy(_θ)
    ps = Flux.params(θ)

    if _data == DEFAULT_DATA && maxiters == typemax(Int)
        error(
            "A data iterator must be provided or the `maxiters` keyword argument must be set.",
        )
    elseif _data == DEFAULT_DATA && maxiters != typemax(Int)
        data = Iterators.repeated((), maxiters)
    elseif maxiters != typemax(Int)
        data = Iterators.take(_data, maxiters) # an iterator that generates at most the first n elements of iter
    else
        data = _data
    end

    t0 = time()

    local x , min_err
    min_err = typemax(eltype(θ)) # dummy variables
    min_opt = 1

    anim = @animate for (i, d) in enumerate(data) # Iterator that yields (i, d) where i is a counter starting at 1, 
        # and d is the d-th value from dataset 
        # calculate gradients of loss function
        # gradient() requires two args: f and params                                       
        gs = Flux.Zygote.gradient(ps) do
            x = loss(θ, d...)
            first(x)
        end
        cb_call = cb(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error(
                "The callback function should return a boolean to decide whether stopping the optimization process.",
            )
        elseif cb_call # stop
            break
        end

        DiffEqFlux.update!(opt, ps, gs)
        if save_best
            if first(x) < first(min_err) # we've found a better solution
                min_opt = opt
                min_err = x
            end
            if i == maxiters  # for last iteration, revert to best
                opt = min_opt
                cb(θ, min_err...)
            end
        end
    end

    t1 = time()

    res = Optim.MultivariateOptimizationResults(
        opt,
        _θ, # initial_x,
        θ, # pick_best_x(f_incr_pick, state),
        first(x), # pick_best_f(f_incr_pick, state, d),
        maxiters, # iteration,
        maxiters >= maxiters, # iteration == options.iterations,
        false, # x_converged,
        0.0, # T(options.x_tol),
        0.0, # T(options.x_tol),
        NaN, # x_abschange(state),
        NaN, # x_abschange(state),
        false, # f_converged,
        0.0, # T(options.f_tol),
        0.0, # T(options.f_tol),
        NaN, # f_abschange(d, state),
        NaN, # f_abschange(d, state),
        false, # g_converged,
        0.0, # T(options.g_tol),
        NaN, # g_residual(d),
        false, # f_increased,
        nothing,
        maxiters,
        maxiters,
        0,
        true,
        NaN,
        t1 - t0,
    )

    return res, anim
end

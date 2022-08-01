# ============================================================
# Training & Optimization Methods
# ============================================================

"""
$(SIGNATURES)

Scientific machine learning trainer

# Arguments
- ``ann``: neural network model
- ``data``: dataset
- ``θ``: parameters of neural network
- ``opt``: optimizer
- ``adtype``: automatical differentiation type
- ``args``: rest arguments
- ``device``: cpu / gpu
- ``maxiters``: maximal iteration number
- ``kwargs``: keyword arguments
"""
function sci_train(
    ann,
    data::Union{Flux.Data.DataLoader,Tuple},
    θ = init_params(ann),
    opt = ADAM(),
    adtype = Optimization.AutoZygote(),
    args...;
    device = cpu,
    maxiters = 200::Integer,
    kwargs...,
)

    data = data |> device
    θ = θ |> device
    L = size(data[1], 2)
    loss(p) = sum(abs2, ann(data[1], p) - data[2]) / L

    cb = function (p, l)
        println("loss: $(loss(p))")
        return false
    end

    return sci_train(
        loss,
        θ,
        opt,
        adtype,
        args...;
        cb = Flux.throttle(cb, 1),
        maxiters = maxiters,
        kwargs...,
    )

end

"""
$(SIGNATURES)
"""
function sci_train(
    loss,
    θ,
    opt = OptimizationPolyalgorithms.PolyOpt(),
    adtype = nothing,
    args...;
    lower_bounds = nothing,
    upper_bounds = nothing,
    cb = nothing,
    callback = (args...) -> (false),
    maxiters = nothing,
    kwargs...,
)

    if adtype === nothing
        if length(θ) < 50
            fdtime = try
                ForwardDiff.gradient(x -> first(loss(x)), θ)
                @elapsed ForwardDiff.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end
            zytime = try
                Zygote.gradient(x -> first(loss(x)), θ)
                @elapsed Zygote.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end

            if fdtime == zytime == Inf
                @warn "AD methods failed, using numerical differentiation. To debug, try ForwardDiff.gradient(loss, θ) or Zygote.gradient(loss, θ)"
                adtype = Optimization.AutoFiniteDiff()
            elseif fdtime < zytime
                adtype = Optimization.AutoForwardDiff()
            else
                adtype = Optimization.AutoZygote()
            end

        else
            adtype = Optimization.AutoZygote()
        end
    end
    if !isnothing(cb)
        callback = cb
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ;
        lb = lower_bounds,
        ub = upper_bounds,
        kwargs...,
    )
    if maxiters !== nothing
        Optimization.solve(optprob, opt, args...; maxiters, callback = callback, kwargs...)
    else
        Optimization.solve(optprob, opt, args...; callback = callback, kwargs...)
    end
end

sci_train(loss, p::NamedTuple, args...; kwargs...) = sci_train(loss, Lux.ComponentArray(p), args...; kwargs...)

"""
$(SIGNATURES)

Scientific machine learning trainer

# Arguments
- ``ann``: neural network model
- ``data``: tuple (X, Y) of dataset
- ``opt``: optimizer 
- ``epoch``: epoch number
- ``batch``: batch size
- ``device``: cpu / gpu
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
function sci_train!(
    ann::PyObject,
    data::Tuple;
    device = cpu,
    split = 0.0,
    epoch = 1,
    batch = 64,
    verbose = 1,
)
    X, Y = data
    ann.fit(
        X,
        Y,
        validation_split = split,
        epochs = epoch,
        batch_size = batch,
        verbose = verbose,
    )

    return nothing
end

# ============================================================
# Training & Optimization Methods
# ============================================================

"""
$(SIGNATURES)

Flux's previous @epochs macro
"""
macro epochs(n, ex)
    :(@progress for i = 1:$(esc(n))
        @info "Epoch $i"
        $(esc(ex))
    end)
end


"""
$(SIGNATURES)

Scientific machine learning trainer

# Arguments
- ``ann``: neural network model
- ``data``: dataset
- ``θ``: parameters of neural network
- ``opt``: optimizer
- ``ad``: automatical differentiation type
- ``args``: rest arguments
- ``device``: cpu / gpu
- ``iters``: maximal iteration number
- ``kwargs``: keyword arguments
"""
function sci_train(
    ann,
    data::Union{Flux.DataLoader,Tuple},
    θ = init_params(ann),
    opt = Flux.Adam(),
    args...;
    device = Flux.cpu,
    iters = 200::Integer,
    ad = Optimization.AutoZygote(),
    kwargs...,
)

    data = data |> device
    θ = θ |> device
    L = size(data[1], 2)
    loss(p) = sum(abs2, ann(data[1], p) - data[2]) / L

    cb = function (p, l)
        println("loss: $l")
        return false
    end

    return sci_train(
        loss,
        θ,
        opt,
        args...;
        cb = Flux.throttle(cb, 1),
        iters = iters,
        ad = ad,
        kwargs...,
    )

end

function sci_train(
    ann::Lux.AbstractExplicitLayer,
    data::Union{Flux.DataLoader,Tuple},
    ps = setup(ann),
    opt = Flux.Adam(),
    args...;
    device = cpu,
    iters = 200::Integer,
    ad = Optimization.AutoZygote(),
    kwargs...,
)

    data = data |> device
    θ, st = ps
    θ = θ |> device
    L = size(data[1], 2)
    loss(p) = sum(abs2, ann(data[1], p, st)[1] - data[2]) / L

    cb = function (p, l)
        println("loss: $l")
        return false
    end

    return sci_train(
        loss,
        θ,
        opt,
        args...;
        cb = Flux.throttle(cb, 1),
        iters = iters,
        ad = ad,
        kwargs...,
    )

end

"""
$(SIGNATURES)
"""
function sci_train(
    loss,
    θ::AbstractVector,
    opt = OptimizationPolyalgorithms.PolyOpt(),
    args...;
    lower_bounds = nothing,
    upper_bounds = nothing,
    cb = nothing,
    callback = (args...) -> (false),
    iters = nothing,
    ad = nothing,
    epochs = nothing,
    kwargs...,
)

    if ad === nothing
        ad = adapt_adtype(loss, θ)
    end
    if !isnothing(cb)
        callback = cb
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), ad)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ;
        lb = lower_bounds,
        ub = upper_bounds,
        kwargs...,
    )

    if iters !== nothing
        return Optimization.solve(optprob, opt, args...; maxiters = iters, callback = callback, kwargs...)
    else
        return Optimization.solve(optprob, opt, args...; callback = callback, kwargs...)
    end

end

function sci_train(
    loss,
    θ::AbstractVector,
    data::Union{Flux.DataLoader,Tuple},
    opt = OptimizationPolyalgorithms.PolyOpt(),
    args...;
    lower_bounds = nothing,
    upper_bounds = nothing,
    cb = nothing,
    callback = (args...) -> (false),
    iters = nothing,
    ad = nothing,
    epochs = nothing,
    batch = 1,
    shuffle = true,
    kwargs...,
)

    if ad === nothing
        ad = adapt_adtype(loss, θ)
    end
    if !isnothing(cb)
        callback = cb
    end

    optf = Optimization.OptimizationFunction((x, p, α, β) -> loss(x, α, β), ad)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ;
        lb = lower_bounds,
        ub = upper_bounds,
        kwargs...,
    )
    
    dl = begin
        if data isa Tuple
            Flux.DataLoader(data, batchsize = batch, shuffle = shuffle)
        else
            data
        end
    end
    if !isnothing(epochs)
        dl = ncycle(dl, epochs)
    end

    if iters !== nothing
        return Optimization.solve(optprob, opt, dl, args...; maxiters = iters, callback = callback, kwargs...)
    else
        return Optimization.solve(optprob, opt, dl, args...; callback = callback, kwargs...)
    end

end

sci_train(loss, p::NamedTuple, args...; kwargs...) =
    sci_train(loss, ComponentArray(p), args...; kwargs...)

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
function sci_train!(ann, data::Tuple, opt = Flux.Adam(); device = Flux.cpu, epoch = 1, batch = 1)
    X, Y = data |> device
    L = size(X, 2)
    data = Flux.DataLoader((X, Y), batchsize = batch, shuffle = true)# |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    @epochs epoch Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

"""
$(SIGNATURES)
"""
function sci_train!(ann, dl::Flux.DataLoader, opt = Flux.Adam(); device = Flux.cpu, epoch = 1)
    X, Y = dl.data |> device
    L = size(X, 2)
    #dl = dl |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    @epochs epoch Flux.train!(loss, ps, dl, opt, cb = Flux.throttle(cb, 1))

    return nothing
end

"""
$(SIGNATURES)

Trainer for Tensorflow model
"""
function sci_train!(
    ann::PyObject,
    data::Tuple;
    device = Flux.cpu,
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


"""
$(SIGNATURES)

Choose automatic differentiation backend adaptively
"""
function adapt_adtype(loss, θ)
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
            ad = Optimization.AutoFiniteDiff()
        elseif fdtime < zytime
            ad = Optimization.AutoForwardDiff()
        else
            ad = Optimization.AutoZygote()
        end
    else
        ad = Optimization.AutoZygote()
    end

    return ad
end

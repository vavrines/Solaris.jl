"""
$(SIGNATURES)

Scientific machine learning trainer

# Arguments
- ``ann``: neural network model
- ``data``: dataset
- ``θ``: parameters of neural network
- ``opt``: optimizer
- ``args``: rest arguments
- ``device``: cpu / gpu
- ``iters``: maximal iteration number
- ``ad``: automatical differentiation type
- ``batch``: batch size
- ``shuffle``: shuffle data (true) or not (false)
- ``kwargs``: rest keyword arguments
"""
function sci_train(
    ann,
    data::Union{DataLoader,Tuple},
    θ=init_params(ann),
    opt=Adam(),
    args...;
    device=Flux.cpu,
    iters=200::Integer,
    ad=AutoZygote(),
    batch=1,
    shuffle=true,
    kwargs...,
)
    function loss(p, _data)
        batch_x, batch_y = _data
        L = size(batch_x, 2)
        pred = ann(batch_x, p)
        return sum(abs2, pred - batch_y) / L
    end

    cb = function (p, l)
        println("loss: $l")
        return false
    end

    return sci_train(
        loss,
        θ,
        data,
        opt,
        args...;
        device=device,
        cb=Flux.throttle(cb, 1),
        iters=iters,
        ad=ad,
        batch=batch,
        shuffle=shuffle,
        kwargs...,
    )
end

function sci_train(
    ann::Lux.AbstractLuxLayer,
    data::Union{DataLoader,Tuple},
    ps=setup(ann),
    opt=Adam(),
    args...;
    device=Flux.cpu,
    iters=200::Integer,
    ad=AutoZygote(),
    batch=1,
    shuffle=true,
    kwargs...,
)
    ann = ann |> device
    θ, st = ps
    st = st |> device
    model = stateful(ann, st)
    function loss(p, _data)
        batch_x, batch_y = _data
        L = size(batch_x, 2)
        pred = model(batch_x, p)
        return sum(abs2, pred - batch_y) / L
    end

    cb = function (p, l)
        println("loss: $l")
        return false
    end

    return sci_train(
        loss,
        θ,
        data,
        opt,
        args...;
        device=device,
        cb=Flux.throttle(cb, 1),
        iters=iters,
        ad=ad,
        batch=batch,
        shuffle=shuffle,
        kwargs...,
    )
end

"""
$(SIGNATURES)
"""
function sci_train(
    loss,
    θ::AbstractVector,
    opt=Adam(),
    args...;
    device=Flux.cpu,
    lower_bounds=nothing,
    upper_bounds=nothing,
    cb=nothing,
    callback=(args...) -> (false),
    iters=200,
    ad=nothing,
    epochs=nothing,
    kwargs...,
)
    if ad === nothing
        ad = adapt_adtype(loss, θ)
    end
    if !isnothing(cb)
        callback = cb
    end
    θ = θ |> device

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), ad)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ;
        lb=lower_bounds,
        ub=upper_bounds,
        kwargs...,
    )

    return Optimization.solve(
        optprob,
        opt,
        args...;
        maxiters=iters,
        callback=callback,
        kwargs...,
    )
end

function sci_train(
    loss,
    θ::AbstractVector,
    data::Union{DataLoader,Tuple},
    opt=Adam(),
    args...;
    device=Flux.cpu,
    lower_bounds=nothing,
    upper_bounds=nothing,
    cb=nothing,
    callback=(args...) -> (false),
    iters=200,
    ad=nothing,
    epochs=nothing,
    batch=1,
    shuffle=true,
    kwargs...,
)
    if ad === nothing
        ad = adapt_adtype(loss, θ)
    end
    if !isnothing(cb)
        callback = cb
    end

    dl = begin
        if data isa Tuple
            DataLoader(data; batchsize=batch, shuffle=shuffle)
        else
            data
        end
    end
    dl = dl |> device
    θ = θ |> device

    optf = Optimization.OptimizationFunction(loss, ad)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ,
        dl;
        lb=lower_bounds,
        ub=upper_bounds,
        kwargs...,
    )

    return Optimization.solve(
        optprob,
        opt,
        args...;
        maxiters=iters,
        callback=callback,
        epochs=epochs,
        kwargs...,
    )
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
function sci_train!(ann, data::Tuple, opt=Adam(); device=Flux.cpu, epoch=1, batch=1)
    X, Y = data |> device
    L = size(X, 2)
    data = DataLoader((X, Y); batchsize=batch, shuffle=true)# |> device

    ann = device(ann)
    loss(m, x, y) = sum(abs2, m(x) - y) / L
    opt_state = Flux.setup(opt, ann)

    @epochs epoch Flux.train!(loss, ann, data, opt_state)

    return nothing
end

"""
$(SIGNATURES)
"""
function sci_train!(ann, dl::DataLoader, opt=Adam(); device=Flux.cpu, epoch=1)
    X, Y = dl.data |> device
    L = size(X, 2)

    ann = device(ann)
    loss(m, x, y) = sum(abs2, m(x) - y) / L
    opt_state = Flux.setup(opt, ann)

    @epochs epoch Flux.train!(loss, ann, dl, opt_state)

    return nothing
end

"""
$(SIGNATURES)

Trainer for Tensorflow model
"""
function sci_train!(
    ann::PyObject,
    data::Tuple;
    device=Flux.cpu,
    split=0.0,
    epoch=1,
    batch=64,
    verbose=1,
)
    X, Y = data
    ann.fit(X, Y; validation_split=split, epochs=epoch, batch_size=batch, verbose=verbose)

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
            ad = AutoFiniteDiff()
        elseif fdtime < zytime
            ad = AutoForwardDiff()
        else
            ad = AutoZygote()
        end
    else
        ad = AutoZygote()
    end

    return ad
end

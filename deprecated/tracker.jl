# ============================================================
# Tracker Methods (see TrackerFlux.jl for details)
# ============================================================

track(m) = fmap(x -> x isa AbstractArray ? Tracker.param(x) : x, m)

untrack(m) = fmap(Tracker.data, m)

function Flux.Optimise.update!(opt, xs::Tracker.Params, gs)
    for x in xs
        Flux.Optimise.update!(opt, x, gs[x])
    end
end

function tracker_mode()
    @eval Flux.Optimise.update!(opt, x, x̄) =
        Tracker.update!(x, -Flux.Optimise.apply!(opt, Tracker.data(x), Tracker.data(x̄)))
    @eval Flux.gradient(f, args...) = Tracker.gradient(f, args...)
end

function zygote_mode()
    @eval Flux.Optimise.update!(opt, x, x̄) =
        Flux.Optimise.update!(x, -Flux.Optimise.apply!(opt, x, x̄))
    @eval Flux.gradient(f, args...) = Flux.Zygote.gradient(f, args...)
end

@testset "Widget" begin
    x = randn(Float32, 10, 2)
    y = rand(Float32, 1, 2)

    model = Dense(10, 1) |> Solaris.track

    function loss(x, y)
        xs = Flux.unstack(x, 3)
        ys = Flux.unstack(y, 3)
        ŷs = model.(xs)
        l = 0.0f0
        for t = 1:length(ŷs)
            l += Flux.mse(ys[t], ŷs[t])
        end
        return l / length(ŷs)
    end
    ps = Flux.params(model)
    data = repeat([(x, y)], 10)
    opt = Adam()
    cb = () -> Flux.reset!(model)
    Solaris.tracker_mode()
    Flux.train!(loss, ps, data, opt, cb = cb)

    ps1 = Flux.params(model)
    Flux.train!(loss, ps1, data, opt, cb = cb)

    model |> Solaris.untrack
    Solaris.zygote_mode()
end

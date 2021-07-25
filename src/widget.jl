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

#=@testset "Widget" begin
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
=#

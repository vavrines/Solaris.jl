using Solaris, Test, Flux

a = device(rand(2))
device(a)

x = randn(Float32, 10, 2)
y = rand(Float32, 1, 2)

model = Dense(10, 1) |> Solaris.track

function loss(x, y)
    xs = Flux.unstack(x, 3)
    ys = Flux.unstack(y, 3)
    ŷs = model.(xs)
    l = 0f0
    for t in 1:length(ŷs)
        l += Flux.mse(ys[t], ŷs[t])
    end
    return l / length(ŷs)
end
ps = Flux.params(model)
data = repeat([(x, y)], 10)
opt = ADAM()
cb = () -> Flux.reset!(model)
Solaris.tracker_mode()
Flux.train!(loss, ps, data, opt, cb = cb)

model |> Solaris.untrack
Solaris.zygote_mode()

include("test_io.jl")
include("test_layer.jl")

# ============================================================
# Deprecated trainer in Flux
# ============================================================

function sci_train!(ann, data::Tuple, opt=Adam(); device=Flux.cpu, epoch=1, batch=1)
    X, Y = data |> device
    L = size(X, 2)
    data = DataLoader((X, Y); batchsize=batch, shuffle=true)# |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    @epochs epoch Flux.train!(loss, ps, data, opt, cb=Flux.throttle(cb, 1))

    return nothing
end

function sci_train!(ann, dl::DataLoader, opt=Adam(); device=Flux.cpu, epoch=1)
    X, Y = dl.data |> device
    L = size(X, 2)
    #dl = dl |> device

    ann = device(ann)
    ps = Flux.params(ann)
    loss(x, y) = sum(abs2, ann(x) - y) / L
    cb = () -> println("loss: $(loss(X, Y))")

    @epochs epoch Flux.train!(loss, ps, dl, opt, cb=Flux.throttle(cb, 1))

    return nothing
end

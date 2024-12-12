model = NeuralOperators.DeepONet(;
    branch=(64, 64),
    trunk=(2, 64),
    branch_activation=Lux.gelu,
    trunk_activation=Lux.gelu,
)

ps, st = Solaris.setup(model)
u = randn(64, 200)
y = randn(2, 10)

Solaris.infer_deeponet(model, ps, st, u, y)

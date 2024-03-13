using OrdinaryDiffEq, SciMLSensitivity, Solaris, Plots
using Optimisers: Adam
using Optimization: AutoZygote

u0 = [2.0; 0.0]
datasize = 30
tspan = (0.0, 1.5)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
t = range(tspan[1], tspan[2], length = datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat = t))

nn = FnChain(FnDense(2, 50, tanh), FnDense(50, 2))
dudt(u, p, t) = nn(u, p)
prob = ODEProblem(dudt, u0, tspan)
p0 = init_params(nn)

function loss(θ)
    pred = solve(prob, Tsit5(), u0 = u0, p = θ, saveat = t) |> Array
    loss = sum(abs2, ode_data .- pred)

    return loss, pred
end

cb = function (p, l, pred)
    println("loss: $(l)")

    pl = scatter(
        t,
        ode_data[1, :],
        label = "truth",
        xlims = [-0.05, 1.55],
        ylims = [-2, 2.1],
    )
    scatter!(pl, t, pred[1, :], label = "prediction")
    display(plot(pl))

    return false
end

res = sci_train(loss, p0, Adam(0.05); cb = cb, iters = 1000, ad = AutoZygote())

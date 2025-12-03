using OrdinaryDiffEq, SciMLSensitivity, Solaris, Lux, Plots

u0 = [2.0; 0.0]
datasize = 30
tspan = (0.0, 1.5)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    return du .= ((u .^ 3)'true_A)'
end
t = range(tspan[1], tspan[2]; length=datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(); saveat=t))

nn = Lux.Chain(x -> x .^ 3, Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
p, st = SR.setup(nn)
p1 = ComponentArray(p)
dudt(x, p, t) = nn(x, p, st) |> first
prob_node = ODEProblem(dudt, u0, tspan)

function loss(θ)
    pred = solve(prob_node, Midpoint(); u0=u0, p=θ, saveat=t) |> Array
    loss = sum(abs2, ode_data .- pred)

    return loss
end

cb = function (p, l)
    display(l)
    return false
end

res = sci_train(loss, p, Adam(0.05); ad=AutoZygote(), callback=cb, maxiters=300)
res = sci_train(loss, res.u, LBFGS(); ad=AutoZygote(), callback=cb, maxiters=100)

sol = solve(prob_node, Midpoint(); u0=u0, p=res.u, saveat=t)
nde_data = sol |> Array

plot(ode_data[1, :])
plot!(ode_data[2, :])
plot!(nde_data[1, :]; line=:dash)
plot!(nde_data[2, :]; line=:dash)

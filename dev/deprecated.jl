DiffEqFlux.paramlength(f::FastConvex) = f.out * (f.zin + f.xin + 1)
DiffEqFlux.initial_params(f::FastConvex) = f.initial_params()

param_length(f::DiffEqFlux.FastDense) = f.out * (f.in + f.bias)
init_params(f::DiffEqFlux.FastDense) = f.initial_params()

DiffEqFlux.initial_params(c::FastICNN) = vcat(initial_params.(c.layers)...)

function (m::FastICNN)(x::AbstractArray, p)
    z = m.layers[1](x, p[1:DiffEqFlux.paramlength(m.layers[1])])
    counter = DiffEqFlux.paramlength(m.layers[1])
    for i = 2:length(m.layers)
        z = m.layers[i](z, x, p[counter+1:counter+DiffEqFlux.paramlength(m.layers[i])])
        counter += DiffEqFlux.paramlength(m.layers[i])
    end

    return z
end

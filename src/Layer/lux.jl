"""
$(SIGNATURES)
"""
init_params(m::Lux.AbstractLuxLayer) = Lux.initialparameters(Random.default_rng(), m)

"""
$(SIGNATURES)
"""
param_length(m::Lux.AbstractLuxLayer) = length(init_params(m))

"""
$(SIGNATURES)

Shortcut for Lux.setup method
"""
setup(x::Lux.AbstractLuxLayer) = Lux.setup(Random.default_rng(), x)

"""
$(SIGNATURES)
"""
apply(m::Lux.AbstractLuxLayer, x, p, st) = m(x, p, st)[1]

"""
$(SIGNATURES)

Transform to Lux.StatefulLuxLayer
"""
function stateful(nn::Lux.AbstractLuxLayer, st; fix=true)
    return Lux.StatefulLuxLayer{fix}(nn, nothing, st)
end

function stateful(nn::Lux.AbstractLuxLayer, ps, st; fix=true)
    return Lux.StatefulLuxLayer{fix}(nn, ps, st)
end

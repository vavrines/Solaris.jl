"""
$(SIGNATURES)
"""
init_params(m::Lux.AbstractExplicitLayer) = Lux.initialparameters(Random.default_rng(), m)

"""
$(SIGNATURES)

Shortcut for Lux.setup method
"""
setup(x::Lux.AbstractExplicitLayer) = Lux.setup(Random.default_rng(), x)

apply(m::Lux.Dense, x, p, st) = m(x, p, st)[1]

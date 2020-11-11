# ============================================================
# Collision Term
# ============================================================

"""
Right-hand side of universal Boltzmann equation
`ube_dfdt(f, p, t)`
* f: particle distribution function in 1D formulation
* p: M, τ, (ann) (parameters)
* t: tspan

"""
function ube_dfdt(f, p, t)
    M, τ, ann = p

    if ann[1] isa FastChain
        df = (M - f) / τ + ann[1](M - f, ann[2])
    elseif ann[1] isa Chain
        df = (M - f) / τ + ann[1](M - f)
    end

    return df
end


"""
Right-hand side of universal Boltzmann equation
`ube_dfdt!(df, f, p, t)`

"""
function ube_dfdt!(df, f, p, t)
    M, τ, ann = p

    if ann[1] isa FastChain
        df .= (M - f) / τ + ann[1](M - f, ann[2])
    elseif ann[1] isa Chain
        df .= (M - f) / τ + ann[1](M - f)
    end
end



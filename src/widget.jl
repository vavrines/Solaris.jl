"""
$(SIGNATURES)

Flux's @epochs macro
"""
macro epochs(n, ex)
    :(@progress for i = 1:$(esc(n))
        @info "Epoch $i"
        $(esc(ex))
    end)
end


"""
$(SIGNATURES)

Transform vector to NamedTuple

# Arguments
- ``v``: vector of parameters
- ``ps``: NamedTuple template of parameters
"""
function vector_nametuple(v::AbstractVector, ps::NamedTuple)
    @assert length(v) == Lux.parameterlength(ps)

    i = 1
    function get_ps(x)
        z = reshape(view(v, i:(i+length(x)-1)), size(x))
        i += length(x)
        return z
    end

    return fmap(get_ps, ps)
end

vector_nametuple(x::ComponentArray) = NamedTuple(x)


"""
$(SIGNATURES)

Transform NamedTuple to ComponentArray

# Arguments
- ``ps``: NamedTuple of parameters
"""
nametuple_carray(ps::NamedTuple) = ComponentArray(ps)


"""
$(SIGNATURES)

Transform NamedTuple to vector

# Arguments
- ``ps``: NamedTuple of parameters
"""
nametuple_vector(ps::NamedTuple) = Vector(nametuple_carray(ps))


"""
$(SIGNATURES)

Default callback function for Optimization solver
"""
function default_callback(θ, l)
    println("loss: $l")
    return false
end


"""
$(SIGNATURES)

Convert data to CPU
"""
function cpu(x...)
    dev = Lux.cpu_device()
    return dev(x...)
end


"""
$(SIGNATURES)

Convert data to GPU
"""
function gpu(x...)
    dev = Lux.gpu_device()
    return dev(x...)
end

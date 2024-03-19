"""
$(SIGNATURES)

Load dataset from file
"""
function load_data(file; mode = :csv, dlm = nothing)
    if mode == :csv
        dataset = File(file; delim = dlm) |> DataFrame
    end

    return dataset
end


"""
$(SIGNATURES)

Load the trained machine learning model
"""
function load_model(file::T; kwargs...) where {T<:AbstractString}
    if file[end-3:end] == "jld2"
        nn = load_model(file, :jld)
    else
        nn = load_model(file, :tf)
    end

    return nn
end

"""
$(SIGNATURES)
"""
function load_model(file::T, mode) where {T<:AbstractString}
    if mode == :jld
        JLD2.@load file nn
    elseif mode == :tf
        copy!(tf, pyimport("tensorflow"))
        nn = tf.keras.models.load_model(file)
    end

    return nn
end


"""
$(SIGNATURES)

Save the trained machine learning model
"""
function save_model(nn; mode = :jld)
    if mode == :jld
        JLD2.@save "model.jld2" nn
    end
end

"""
$(SIGNATURES)
"""
function save_model(nn::PyObject)
    nn.save("model.h5")
end

"""
Affine layer of neural network

@args: w, b, σ

"""
struct Affine{T1,T2}
    w::T1
    b::T2
    σ::Function

    function Affine(
        W::AbstractArray,
        B::AbstractArray,
        FUNC=identity::Function,
    )
        return new{typeof(W),typeof(B)}(W, B, FUNC)
    end

    function Affine(
        in::Integer,
        out::Integer,
        σ = identity::Function;
        fw = randn::Function,
        fb = zeros::Function,
        isBias = true::Bool,
    )
        if isBias
            return Affine(fw(Float32, out, in), fb(Float32, out), σ)
        else
            return Affine(fw(Float32, out, in), Flux.Zeros(out), σ)
        end
    end
end

# ------------------------------------------------------------
# Functor
# ------------------------------------------------------------
Flux.@functor Affine # work with Flux.params

(L::Affine)(x) = L.w * x .+ L.b

"""
avoid hitting generic matmul in simple cases
Base.matmul is slow so it's worth the extra conversion to hit BLAS 
"""
(a::Affine{<:Any,W})(x::AbstractArray{T}) where {T<:Union{Float32,Float64}, W<:AbstractArray{T}} =
    invoke(a, Tuple{AbstractArray}, x)

(a::Affine{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T<:Union{Float32,Float64}, W<:AbstractArray{T}} =
    a(T.(x))
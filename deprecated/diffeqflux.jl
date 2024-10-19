"""
Dated dense layer in DiffEqFlux

"""

using ZygoteRules: @adjoint

isgpu(x) = false
ifgpufree(x) = nothing

isgpu(::CUDA.CuArray) = true
isgpu(::Transpose{<:Any,<:CUDA.CuArray}) = true
isgpu(::Adjoint{<:Any,<:CUDA.CuArray}) = true
ifgpufree(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
ifgpufree(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)
ifgpufree(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)

#const TrackedArray = Tracker.TrackedArray
#isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
#isgpu(::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = true
#isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
#ifgpufree(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.data)
#ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) =
#    CUDA.unsafe_free!((x.data).parent)
#ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) =
#    CUDA.unsafe_free!((x.data).parent)

"""
$(TYPEDEF)

Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine function.
Initial parameters are taken to match `Flux.Dense`. 'bias' represents b in the layer and 
it defaults to true. 'precache' is used to preallocate memory for the intermediate variables 
calculated during each pass. This avoids heap allocations in each pass which would otherwise 
slow down the computation, it defaults to false. This function has specializations on `tanh` 
for a slightly faster adjoint with Zygote.

# Fields

$(FIELDS)
"""
struct FastDense{F,F2,C} <: AbstractExplicitLayer
    out::Int
    in::Int
    σ::F
    initial_params::F2
    cache::C
    bias::Bool
    numcols::Int

    function FastDense(
        in::Integer,
        out::Integer,
        σ=identity;
        bias=true,
        numcols=1,
        precache=false,
        initW=Flux.glorot_uniform,
        initb=Flux.zeros32,
    )
        temp = ((bias == false) ? vcat(vec(initW(out, in))) :
         vcat(vec(initW(out, in)), initb(out)))
        initial_params() = temp
        if precache == true
            cache = (
                cols=zeros(Int, 1),
                W=zeros(out, in),
                y=zeros(out, numcols),
                yvec=zeros(out),
                r=zeros(out, numcols),
                zbar=zeros(out, numcols),
                Wbar=zeros(out, in),
                xbar=zeros(in, numcols),
                pbar=if bias == true
                    zeros((out * in) + out)
                else
                    zeros(out * in)
                end,
            )
        else
            cache = nothing
        end
        _σ = NNlib.fast_act(σ)
        return new{typeof(_σ),typeof(initial_params),typeof(cache)}(
            out,
            in,
            _σ,
            initial_params,
            cache,
            bias,
            numcols,
        )
    end
end

(f::FastDense)(x::Number, p) = ((f.bias == true) ?
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x)))

@adjoint function (f::FastDense)(x::Number, p)
    if typeof(f.cache) <: Nothing
        if !isgpu(p)
            W = @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            W = reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        if f.bias == true
            b = p[(f.out*f.in+1):end]
            r = W * x .+ b
            ifgpufree(b)
        else
            r = W * x
        end
        y = f.σ.(r)
    else
        if !isgpu(p)
            f.cache.W .= @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            f.cache.W .= reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        mul!(@view(f.cache.r[:, 1]), f.cache.W, x)
        if f.bias == true
            # @view(f.cache.r[:,1]) .+= @view(p[(f.out*f.in + 1):end])
            b = @view(p[(f.out*f.in+1):end])
            @view(f.cache.r[:, 1]) .+= b
        end
        f.cache.yvec .= f.σ.(@view(f.cache.r[:, 1]))
    end
    function FastDense_adjoint(ȳ)
        if typeof(f.cache) <: Nothing
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                zbar = ȳ .* (1 .- y .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                zbar = ȳ
            else
                zbar = ȳ .* ForwardDiff.derivative.(f.σ, r)
            end
            Wbar = zbar * x'
            bbar = zbar
            xbar = W' * zbar
            pbar = if f.bias == true
                tmp =
                    typeof(bbar) <: AbstractVector ? vec(vcat(vec(Wbar), bbar)) :
                    vec(vcat(vec(Wbar), sum(bbar; dims=2)))
                tmp
            else
                vec(Wbar)
            end
            ifgpufree(Wbar)
            ifgpufree(r)
            xb = xbar[1, 1]
            nothing, xb, pbar
        else
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                @view(f.cache.zbar[:, 1]) .= ȳ .* (1 .- (f.cache.yvec) .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                @view(f.cache.zbar[:, 1]) .= ȳ
            else
                @view(f.cache.zbar[:, 1]) .=
                    ȳ .* ForwardDiff.derivative.(f.σ, @view(f.cache.r[:, 1]))
            end
            mul!(f.cache.Wbar, @view(f.cache.zbar[:, 1]), x')
            mul!(@view(f.cache.xbar[:, 1]), f.cache.W', @view(f.cache.zbar[:, 1]))
            f.cache.pbar .= if f.bias == true
                vec(vcat(vec(f.cache.Wbar), @view(f.cache.zbar[:, 1])))# bbar = zbar
            else
                vec(f.cache.Wbar)
            end
            xbar = f.cache.xbar[1, 1]
            nothing, xbar, f.cache.pbar
        end
    end
    if typeof(f.cache) <: Nothing
        y, FastDense_adjoint
    else
        f.cache.yvec, FastDense_adjoint
    end
end

(f::FastDense)(x::AbstractVector, p) = ((f.bias == true) ?
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x)))

@adjoint function (f::FastDense)(x::AbstractVector, p)
    if typeof(f.cache) <: Nothing
        if !isgpu(p)
            W = @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            W = reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        if f.bias == true
            b = p[(f.out*f.in+1):end]
            r = W * x .+ b
            ifgpufree(b)
        else
            r = W * x
        end
        y = f.σ.(r)
    else
        if !isgpu(p)
            f.cache.W .= @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            f.cache.W .= reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        mul!(@view(f.cache.r[:, 1]), f.cache.W, x)
        if f.bias == true
            # @view(f.cache.r[:,1]) .+= @view(p[(f.out*f.in + 1):end])
            b = @view(p[(f.out*f.in+1):end])
            @view(f.cache.r[:, 1]) .+= b
        end
        f.cache.yvec .= f.σ.(@view(f.cache.r[:, 1]))
    end
    function FastDense_adjoint(ȳ)
        if typeof(f.cache) <: Nothing
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                zbar = ȳ .* (1 .- y .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                zbar = ȳ
            else
                zbar = ȳ .* ForwardDiff.derivative.(f.σ, r)
            end
            Wbar = zbar * x'
            bbar = zbar
            xbar = W' * zbar
            pbar = if f.bias == true
                tmp =
                    typeof(bbar) <: AbstractVector ? vec(vcat(vec(Wbar), bbar)) :
                    vec(vcat(vec(Wbar), sum(bbar; dims=2)))
                !(typeof(f.σ) <: typeof(identity)) && ifgpufree(bbar)
                tmp
            else
                vec(Wbar)
            end
            ifgpufree(Wbar)
            ifgpufree(r)
            nothing, xbar, pbar
        else
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                @view(f.cache.zbar[:, 1]) .= ȳ .* (1 .- (f.cache.yvec) .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                @view(f.cache.zbar[:, 1]) .= ȳ
            else
                @view(f.cache.zbar[:, 1]) .=
                    ȳ .* ForwardDiff.derivative.(f.σ, @view(f.cache.r[:, 1]))
            end
            mul!(f.cache.Wbar, @view(f.cache.zbar[:, 1]), x')
            mul!(@view(f.cache.xbar[:, 1]), f.cache.W', @view(f.cache.zbar[:, 1]))
            f.cache.pbar .= if f.bias == true
                vec(vcat(vec(f.cache.Wbar), @view(f.cache.zbar[:, 1])))# bbar = zbar
            else
                vec(f.cache.Wbar)
            end
            nothing, @view(f.cache.xbar[:, 1]), f.cache.pbar
        end
    end
    if typeof(f.cache) <: Nothing
        y, FastDense_adjoint
    else
        f.cache.yvec, FastDense_adjoint
    end
end

(f::FastDense)(x::AbstractMatrix, p) = ((f.bias == true) ?
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x .+ p[(f.out*f.in+1):end])) :
 (f.σ.(reshape(p[1:(f.out*f.in)], f.out, f.in) * x)))

@adjoint function (f::FastDense)(x::AbstractMatrix, p)
    if typeof(f.cache) <: Nothing
        if !isgpu(p)
            W = @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            W = reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        if f.bias == true
            b = p[(f.out*f.in+1):end]
            r = W * x .+ b
            ifgpufree(b)
        else
            r = W * x
        end
        y = f.σ.(r)
    else
        if !isgpu(p)
            f.cache.W .= @view(p[reshape(1:(f.out*f.in), f.out, f.in)])
        else
            f.cache.W .= reshape(@view(p[1:(f.out*f.in)]), f.out, f.in)
        end
        f.cache.cols[1] = size(x)[2]
        mul!(@view(f.cache.r[:, 1:f.cache.cols[1]]), f.cache.W, x)
        if f.bias == true
            @view(f.cache.r[:, 1:f.cache.cols[1]]) .+= @view(p[(f.out*f.in+1):end])
        end
        @view(f.cache.y[:, 1:f.cache.cols[1]]) .=
            f.σ.(@view(f.cache.r[:, 1:f.cache.cols[1]]))
    end
    function FastDense_adjoint(ȳ)
        if typeof(f.cache) <: Nothing
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                zbar = ȳ .* (1 .- y .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                zbar = ȳ
            else
                zbar = ȳ .* ForwardDiff.derivative.(f.σ, r)
            end
            Wbar = zbar * x'
            bbar = zbar
            xbar = W' * zbar
            pbar = if f.bias == true
                tmp =
                    typeof(bbar) <: AbstractVector ? vec(vcat(vec(Wbar), bbar)) :
                    vec(vcat(vec(Wbar), sum(bbar; dims=2)))
                !(typeof(f.σ) <: typeof(identity)) && ifgpufree(bbar)
                tmp
            else
                vec(Wbar)
            end
            ifgpufree(Wbar)
            ifgpufree(r)
            nothing, xbar, pbar
        else
            if typeof(f.σ) <: typeof(NNlib.tanh_fast)
                @view(f.cache.zbar[:, 1:f.cache.cols[1]]) .=
                    ȳ .* (1 .- @view(f.cache.y[:, 1:f.cache.cols[1]]) .^ 2)
            elseif typeof(f.σ) <: typeof(identity)
                @view(f.cache.zbar[:, 1:f.cache.cols[1]]) .= ȳ
            else
                @view(f.cache.zbar[:, 1:f.cache.cols[1]]) .=
                    ȳ .*
                    ForwardDiff.derivative.(f.σ, @view(f.cache.r[:, 1:f.cache.cols[1]]))
            end
            mul!(f.cache.Wbar, @view(f.cache.zbar[:, 1:f.cache.cols[1]]), x')
            mul!(
                @view(f.cache.xbar[:, 1:f.cache.cols[1]]),
                f.cache.W',
                @view(f.cache.zbar[:, 1:f.cache.cols[1]])
            )
            f.cache.pbar .= if f.bias == true
                vec(vcat(
                    vec(f.cache.Wbar),
                    sum(@view(f.cache.zbar[:, 1:f.cache.cols[1]]); dims=2),
                ),)# bbar = zbar
            else
                vec(f.cache.Wbar)
            end
            nothing, @view(f.cache.xbar[:, 1:f.cache.cols[1]]), f.cache.pbar
        end
    end
    if typeof(f.cache) <: Nothing
        y, FastDense_adjoint
    elseif f.numcols == f.cache.cols[1]
        f.cache.y, FastDense_adjoint
    else
        @view(f.cache.y[:, 1:f.cache.cols[1]]), FastDense_adjoint
    end
end

param_length(f::FastDense) = f.out * (f.in + f.bias)

init_params(f::FastDense) = f.initial_params()

apply(m::FastDense, x, p) = m(x, p)

module Solaris

using CSV
using CUDA: CuArray
using DataFrames
using DiffEqFlux
using ForwardDiff
using DiffEqFlux.Zygote
using DocStringExtensions
using Flux
using JLD2
using Lux
using Optim
using Optimization
using PyCall
import Tracker

export device
export AbstractLayer, AbstractChain, Shortcut, Convex, ICNN, FastConvex, FastICNN
export sci_train, sci_train!, vis_train

include("widget.jl")
include("io.jl")
include("Layer/layer.jl")
include("train.jl")

const tf = PyNULL()

end

module Solaris

using CSV
using CUDA: CuArray
using DataFrames
using DiffEqFlux
using ForwardDiff
using DocStringExtensions
using Flux
using JLD2
using Optim
using Optimization
using PyCall
using Zygote
import Lux
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

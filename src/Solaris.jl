module Solaris

using CSV
using CUDA: CuArray
using DataFrames
using DiffEqFlux
using Flux
using Optim
using Plots
using PyCall
import BSON
import JLD2
import Tracker

export device
export AbstractLayer,
       AbstractChain,
       Shortcut,
       Convex,
       ICNN,
       FastConvex,
       FastICNN
export vis_train

include("widget.jl")
include("io.jl")
include("Layer/layer.jl")
include("train.jl")

const tf = PyNULL()

end

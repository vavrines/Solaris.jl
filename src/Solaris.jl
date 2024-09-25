module Solaris

using DocStringExtensions
using LinearAlgebra
using Optimization
using PyCall
using Random

using CSV: File
using ComponentArrays: ComponentArray
using DataFrames: DataFrame
using Functors: fmap
using IterTools: ncycle
using NNlib: fast_act
using OptimizationOptimisers: Descent, Adam, AdamW
using OptimizationOptimJL: BFGS, LBFGS
using ProgressLogging: @progress

import Flux
import ForwardDiff
import JLD2
import Lux
import Zygote

export SR
export AutoForwardDiff, AutoReverseDiff, AutoTracker, AutoZygote, AutoEnzyme
export Descent, Adam, AdamW, BFGS, LBFGS
export FnDense, FnChain, Shortcut
export Convex, ICNN, FastConvex, FastICNN
export init_params
export load_data, load_model, save_model
export sci_train, sci_train!

include("widget.jl")
include("io.jl")
include("Layer/layer.jl")
include("train.jl")

const tf = PyNULL()
const SR = Solaris

end

module Solaris

using ComponentArrays
using CUDA
using ForwardDiff
using DocStringExtensions
using JLD2
using LinearAlgebra
using NNlib
using Optimization
using OptimizationFlux
using OptimizationOptimJL
using OptimizationPolyalgorithms
using PyCall
using Random
using Zygote

using CSV: File
using DataFrames: DataFrame
using Functors: fmap
using IterTools: ncycle
using ProgressLogging: @progress
using ZygoteRules: @adjoint

import Flux
import Lux

export SR
export AutoForwardDiff, AutoReverseDiff, AutoTracker, AutoZygote, AutoEnzyme
export FnDense, FnChain, Shortcut
export Convex, ICNN, FastConvex, FastICNN
export init_params, param_length, apply
export load_data, load_model, save_model
export sci_train, sci_train!

include("widget.jl")
include("io.jl")
include("Layer/layer.jl")
include("train.jl")

const tf = PyNULL()
const SR = Solaris

end

module Solaris

using ComponentArrays
using CSV
using CUDA
using DataFrames
using ForwardDiff
using DocStringExtensions
using JLD2
using LinearAlgebra
using NNlib
using Optim
using Optimization
using OptimizationFlux
using OptimizationOptimJL
using OptimizationPolyalgorithms
using ProgressLogging
using PyCall
using Random
using Zygote
using ZygoteRules
using IterTools: ncycle
import Flux
import Lux
import Tracker

export SR
export FnDense, FnChain, Shortcut
export Convex, ICNN, FastConvex, FastICNN
export init_params, param_length
export sci_train, sci_train!

include("widget.jl")
include("io.jl")
include("Layer/layer.jl")
include("train.jl")

const tf = PyNULL()
const SR = Solaris

end

module Solaris

using CSV
using CUDA
using DataFrames
using ForwardDiff
using DocStringExtensions
using Flux
using JLD2
using LinearAlgebra
using Optim
using Optimization
using OptimizationFlux
using OptimizationOptimJL
using OptimizationPolyalgorithms
using PyCall
using Random
using Zygote
using ZygoteRules
using IterTools: ncycle
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

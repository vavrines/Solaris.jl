using Solaris, Test
import Flux, Lux, NeuralOperators

cd(@__DIR__)

include("test_io.jl")
include("test_layer.jl")
include("test_infer.jl")
include("test_train.jl")
include("test_widget.jl")

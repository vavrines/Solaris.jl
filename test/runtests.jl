using Solaris, Test, Optim
import Flux, Lux

cd(@__DIR__)

include("test_io.jl")
include("test_layer.jl")
include("test_train.jl")
include("test_widget.jl")

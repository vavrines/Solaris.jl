module Solaris

using Flux
using DiffEqFlux
using Optim
using Plots
import Tracker

export device
export track, untrack, tracker_mode, zygote_mode
export Affine
export vis_train

include("widget.jl")
include("tracker.jl")
include("layer.jl")
include("train.jl")

end # module
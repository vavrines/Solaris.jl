module Solaris

using Kinetic
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using Optim
using FileIO
using JLD2
using Plots
import Tracker

export device
export track, untrack, tracker_mode, zygote_mode
export vis_train
export ube_dfdt, ube_dfdt!
export step_ube!

include("widget.jl")
include("tracker.jl")
include("train.jl")
include("collision.jl")
include("solver.jl")

end # module

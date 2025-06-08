"""
$(SIGNATURES)

Fast inference for DeepONet G(u)(y)

# Arguments

- ``model``: DeepONet model
- ``ps``: parameters for neural network (branch and trunk)
- ``st``: state for neural network
- ``u``: discrete sensored function {u(xᵢ)} with i=1:m
- ``y``: input data

Note that the shape of u and y can be totally different.

Consider 1D advection equation ∂ₜu + a∂ₓu = 0, with the initial condition u(x, 0) = g(x).
Suppose that we are given 200 instances of the IC, discretized at 64 points for each;
and we would like to infer the solution for 10 different points in (x, t).
This makes the branch input of shape (64, 200) and the trunk input of shape (2, 10).

In NeuralOperators.jl, the inputs must be batched, so the branch input is of shape (64, 200, 1)
and the trunk input is of shape (2, 10, 1).
Here we omit this tedious, mind-numbing step as much as possible.
"""
function infer_deeponet(model, ps, st, u, y)
    bs = model.model.layers.layer_1.layers.branch(u, ps.layer_1.branch, st.layer_1.branch)[1]
    ts = model.model.layers.layer_1.layers.trunk(y, ps.layer_1.trunk, st.layer_1.trunk)[1]
    pred = bs * ts

    return pred
end

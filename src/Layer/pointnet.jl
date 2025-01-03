# ------------------------------------------------------------
# https://github.com/FluxML/Flux3D.jl/blob/master/src/models/pointnet.jl
# ------------------------------------------------------------

function conv_bn_block(input, output, kernel_size)
    return Flux.Chain(
        Flux.Conv((kernel_size...,), input => output),
        Flux.BatchNorm(output),
        x -> Flux.relu.(x),
    )
end

stnKD(K::Int) = Flux.Chain(
    Flux.Conv((1,), K => 64, Flux.relu),
    Flux.BatchNorm(64),
    Flux.Conv((1,), 64 => 128, Flux.relu),
    Flux.BatchNorm(128),
    Flux.Conv((1,), 128 => 1024, Flux.relu),
    Flux.BatchNorm(1024),
    x -> maximum(x; dims=1),
    x -> reshape(x, :, size(x, 3)),
    Flux.Dense(1024, 512, Flux.relu),
    Flux.Dense(512, 256, Flux.relu),
    Flux.BatchNorm(256),
    Flux.Dense(256, K * K),
    x -> reshape(x, K, K, size(x, 2)),
    x -> PermutedDimsArray(x, (2, 1, 3)),
)

"""
$(TYPEDEF)

Flux implementation of PointNet classification model.
"""
struct PointNet
    stn::Any
    fstn::Any
    conv_block1::Any
    feat::Any
    cls::Any
end

"""
Construct PointNet model

- `num_classes` - number of classes in the dataset
- `hidden_dims` - Hiddem dimension in PointNet model
"""
function PointNet(num_classes::Int=10, K::Int=64)
    stn = stnKD(3)
    fstn = stnKD(K)
    conv_block1 = conv_bn_block(3, 64, (1,))
    feat = Flux.Chain(
        Flux.Conv((1,), 64 => 128, Flux.relu),
        Flux.BatchNorm(128),
        Flux.Conv((1,), 128 => 1024),
        Flux.BatchNorm(1024),
        x -> maximum(x; dims=1),
        x -> reshape(x, 1024, :),
        Flux.Dense(1024, 512, Flux.relu),
        Flux.BatchNorm(512),
        Flux.Dense(512, 256, Flux.relu),
        Flux.Dropout(0.4),
        Flux.BatchNorm(256),
    )
    cls = Flux.Dense(256, num_classes, Flux.relu)
    return PointNet(stn, fstn, conv_block1, feat, cls)
end

function (m::PointNet)(X)
    # X: [3, N, B]

    X = permutedims(X, (2, 1, 3))
    # X: [N, 3, B]

    X = Flux.batched_mul(X, m.stn(X))
    # X: [3, 3, B]

    X = m.conv_block1(X)
    # X: [3, 64, B]

    X = Flux.batched_mul(X, m.fstn(X))
    # X: [3, 64, B]

    X = m.feat(X)
    # X: [256, B]

    X = m.cls(X)
    # X: [num_classes, B]

    return Flux.softmax(X; dims=1)
end

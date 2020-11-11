"""
Adapt computing architecture

"""
function device(isgpu, args...)
    if isgpu
        return gpu(args)
    else
        return cpu(args)
    end
end
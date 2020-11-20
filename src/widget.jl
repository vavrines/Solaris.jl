"""
Adapt computing architecture

"""
function device(isgpu::Bool, args...)
    if isgpu
        return gpu(args)
    else
        return cpu(args)
    end
end
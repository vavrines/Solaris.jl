"""
$(SIGNATURES)

Flux's @epochs macro
"""
macro epochs(n, ex)
    :(@progress for i = 1:$(esc(n))
        @info "Epoch $i"
        $(esc(ex))
    end)
end

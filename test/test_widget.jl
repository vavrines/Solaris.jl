m = Lux.Dense(2, 2)
ps = SR.init_params(m)
v = SR.nametuple_vector(ps)
SR.vector_nametuple(v, ps)
ps |> SR.cpu

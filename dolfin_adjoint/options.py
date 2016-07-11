from backend import parameters, Parameters

adj_params = Parameters("adjoint")
adj_params.add("record_all", True)
adj_params.add("test_hermitian", False)
adj_params.add("test_derivative", False)
adj_params.add("fussy_replay", True)
adj_params.add("stop_annotating", False)
adj_params.add("cache_factorizations", False)
adj_params.add("debug_cache", False)
adj_params.add("symmetric_bcs", False)
adj_params.add("allow_zero_derivatives", False)

opt_params = Parameters("optimization")
opt_params.add("test_gradient", False)
opt_params.add("test_gradient_seed", 0.0001)

parameters.add(adj_params)
parameters.add(opt_params)

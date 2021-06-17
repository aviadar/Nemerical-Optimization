from Code.src.unconstrained_min import line_search


def interior_pt(func, x0, obj_tol, param_tol, max_inner_loops,
                t=1.0, mu=10.0, epsilon=1e-6, max_outer_loops=100):
    val_hist = []
    x_hist = []
    new_x = x0.copy()
    success = False

    m = float(len(func.inequality_constraints))
    for iteration in range(max_outer_loops):
        func.f0.t = t

        dir_selection_method = 'gd'
        _, last_x, val_hist_temp, x_hist_temp = line_search(func, new_x, obj_tol, param_tol, max_inner_loops,
                                                        dir_selection_method)

        val_hist += val_hist_temp
        x_hist += x_hist_temp
        if m / t < epsilon:
            success = True
            break

        new_x = last_x
        t *= mu

    return success, last_x, val_hist, x_hist
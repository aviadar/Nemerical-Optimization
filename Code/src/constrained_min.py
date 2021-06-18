import numpy as np
from Code.src.unconstrained_min import line_search


def interior_pt(func, x0, obj_tol, param_tol, max_inner_loops,
                t=1.0, mu=10.0, epsilon=1e-6, max_outer_loops=100):
    val_hist = []
    x_hist = None
    new_x = x0.copy()
    success = False

    m = float(len(func.inequality_constraints))
    if func.A is None:
        dir_selection_method = 'nt'
    else:
        dir_selection_method = 'nt_equality'

    for iteration in range(max_outer_loops):
        func.f0.t = t

        _, last_x, _, x_hist_temp = line_search(func, new_x, obj_tol, param_tol, max_inner_loops,
                                                        dir_selection_method)

        # val_hist.append(val_hist_temp)
        if x_hist is None:
            x_hist = x_hist_temp
        else:
            x_hist = np.append(x_hist, x_hist_temp, axis=1)

        if m / t < epsilon:
            success = True
            break

        new_x = last_x
        t *= mu

    func.f0.t = 1.0
    val_hist = [func.f0.evaluate(x) for x in x_hist.T]

    return success, last_x, val_hist, x_hist

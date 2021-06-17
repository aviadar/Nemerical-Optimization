import numpy as np
from src.utils import report


def line_search(f, x0, obj_tol, param_tol, max_iter, dir_selection_method='gd', init_step_len=1.0, slope_ratio=1e-4,
                back_track_factor=0.2):
    prev_x = x0.copy()
    prev_val = f.evaluate(x0)
    val_hist = [prev_val.reshape((1,))]
    x_hist = x0.copy()
    if dir_selection_method == 'bfgs':
        first_iteration = True
        Bk = np.eye(len(x0))
    for curr_iter in range(max_iter):
        if dir_selection_method == 'gd':
            pk = -f.evaluate_grad(prev_x)
        elif dir_selection_method == 'nt':
            pk = newton_dir(f, prev_x)
        elif dir_selection_method == 'bfgs':
            if first_iteration:
                pk, Bk = bfgs_dir(f, x_hist, None, None, first_iteration)
                first_iteration = False
            else:
                pk, Bk = bfgs_dir(f, x_hist[:, -1].reshape(-1, 1), x_hist[:, -2].reshape(-1, 1), Bk, first_iteration)
        else:
            raise Exception("dir_selection_method = [‘gd’, ‘nt’, ‘bfgs’] only!")

        step_size = get_step_size_wolfe(f, prev_x, pk, init_step_len, slope_ratio, back_track_factor)
        new_x = prev_x + step_size * pk
        x_hist = np.append(x_hist, new_x, axis=1)

        new_val = f.evaluate(new_x)
        val_hist.append(new_val.reshape((1,)))

        if abs(prev_val - new_val) < obj_tol or np.linalg.norm(new_x-prev_x) < param_tol:
            report(curr_iter, new_x, new_val, np.linalg.norm(new_x - prev_x), abs(prev_val - new_val))
            return True, new_x, val_hist, x_hist

        report(curr_iter, new_x, new_val, np.linalg.norm(new_x-prev_x), abs(prev_val - new_val))
        prev_x = new_x
        prev_val = new_val

    return False, new_x, val_hist, x_hist


def bfgs_dir(f, xk_1, xk, Bk, first_iteration):
    """ hessian approximation using BFGS """
    if first_iteration:
        Bk_1 = np.eye(len(xk_1))
    else:
        sk = (xk_1-xk).reshape(-1, 1)
        yk = (f.evaluate_grad(xk_1) - f.evaluate_grad(xk)).reshape(-1, 1)
        Bk_1 = Bk - (Bk @ sk @ sk.T @ Bk) /(sk.T @ Bk @ sk) + (yk @ yk.T) / (yk.T @ sk)
    return -(mat_inv(Bk_1) @ f.evaluate_grad(xk_1)).reshape(-1, 1), Bk_1


def newton_dir(f, x):
    """ hessian """
    return -mat_inv(f.evaluate_hess(x)) @ f.evaluate_grad(x)


def mat_inv(A):
    return np.linalg.solve(A, np.eye(A.shape[0]))


def get_step_size_wolfe(f, xk, pk, init_step_len, slope_ratio, back_track_factor):
    alpha = init_step_len
    for i in range(50):
        if f.evaluate(xk + alpha * pk) <= (f.evaluate(xk) + slope_ratio * alpha * f.evaluate_grad(xk).T @ pk):
            return alpha
        else:
            alpha *= back_track_factor
    raise Exception('1st wolfe condition was not applied')

import numpy as np
import unittest
from Code.src.constrained_min import interior_pt
from Code.src.unconstrained_min import line_search
from Code.tests.examples import ConstrainedQuadraticFunction
from Code.src.utils import final_report, plot_contours_paths, plot_val_hist


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        x0 = np.array([0.1, 0.2, 0.7]).reshape(-1, 1)
        obj_tol = 10e-12
        param_tol = 10e-8
        max_inner_loops = 100

        constrained_qp = ConstrainedQuadraticFunction()
        success, last_x, val_hist, x_hist = interior_pt(func=constrained_qp, x0=x0, obj_tol=obj_tol,
                                                        param_tol=param_tol, max_inner_loops=max_inner_loops)




if __name__ == '__main__':
    unittest.main()
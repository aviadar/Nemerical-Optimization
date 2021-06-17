import numpy as np
import unittest
from Code.src.unconstrained_min import line_search
from Code.tests.examples import Q1_quad, Q2_quad, Q3_quad, QuadraticFunction, RosenbrockFunction
from Code.src.utils import final_report, plot_contours_paths, plot_val_hist


class TestUnconstrainedMin(unittest.TestCase):

    def test_quad_min_gd(self):
        x0 = np.array([[1], [1]])
        obj_tol = 10e-12
        param_tol = 10e-8
        max_iter = 100


        ## Qaudratic Functions ##
        ##=====================##
        f_quad_1 = QuadraticFunction(Q1_quad)
        f_quad_2 = QuadraticFunction(Q2_quad)
        f_quad_3 = QuadraticFunction(Q3_quad)

        dir_selection_method = 'gd'

        success, last_x, val_hist, x_hist = line_search(f_quad_1, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'GD 1st Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_1, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_2, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'GD 2nd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_2, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_3, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'GD 3rd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_3, x_hist)

    def test_quad_min_nt(self):
        x0 = np.array([[1], [1]])
        obj_tol = 10e-12
        param_tol = 10e-8
        max_iter = 100

        ## Qaudratic Functions ##
        ##=====================##
        f_quad_1 = QuadraticFunction(Q1_quad)
        f_quad_2 = QuadraticFunction(Q2_quad)
        f_quad_3 = QuadraticFunction(Q3_quad)

        dir_selection_method = 'nt'

        success, last_x, val_hist, x_hist = line_search(f_quad_1, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'NT 1st Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_1, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_2, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'NT 2nd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_2, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_3, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'NT 3rd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_3, x_hist)

    def test_quad_min_bfgs(self):
        x0 = np.array([[1], [1]])
        obj_tol = 10e-12
        param_tol = 10e-8
        max_iter = 100

        ## Qaudratic Functions ##
        ##=====================##
        f_quad_1 = QuadraticFunction(Q1_quad)
        f_quad_2 = QuadraticFunction(Q2_quad)
        f_quad_3 = QuadraticFunction(Q3_quad)

        dir_selection_method = 'bfgs'

        success, last_x, val_hist, x_hist = line_search(f_quad_1, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'BFGS 1st Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_1, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_2, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'BFGS 2nd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_2, x_hist)

        success, last_x, val_hist, x_hist = line_search(f_quad_3, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'BFGS 3rd Qaudratic objective function vs. iterations')
        plot_contours_paths(f_quad_3, x_hist)

    def test_rosenbrock_min_gd(self):

        x0 = np.array([[2], [2]])
        obj_tol = 10e-7
        param_tol = 10e-8
        max_iter = 10000

        f_rosenbrock = RosenbrockFunction()
        dir_selection_method = 'gd'

        success, last_x, val_hist, x_hist = line_search(f_rosenbrock, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'GD rosenbrock objective function vs. iterations')
        plot_contours_paths(f_rosenbrock, x_hist)

    def test_rosenbrock_min_nt(self):
        x0 = np.array([[2], [2]])
        obj_tol = 10e-7
        param_tol = 10e-8
        max_iter = 10000

        f_rosenbrock = RosenbrockFunction()
        dir_selection_method = 'nt'

        success, last_x, val_hist, x_hist = line_search(f_rosenbrock, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'NT rosenbrock objective function vs. iterations')
        plot_contours_paths(f_rosenbrock, x_hist)

    def test_rosenbrock_min_bfgs(self):
        x0 = np.array([[2], [2]])
        obj_tol = 10e-7
        param_tol = 10e-8
        max_iter = 10000

        f_rosenbrock = RosenbrockFunction()
        dir_selection_method = 'bfgs'

        success, last_x, val_hist, x_hist = line_search(f_rosenbrock, x0, obj_tol, param_tol, max_iter,
                                                        dir_selection_method)
        final_report(success, last_x)
        plot_val_hist(val_hist, 'BFGS rosenbrock objective function vs. iterations')
        plot_contours_paths(f_rosenbrock, x_hist)


if __name__ == '__main__':
    unittest.main()

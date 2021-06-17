import numpy as np
from abc import ABC, abstractmethod

Q1_quad = np.array([[1, 0], [0, 1]])
Q2_quad = np.array([[5, 0], [0, 1]])
Q3_quad = Q = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]]).transpose() @ np.array(
    [[5, 0], [0, 1]]) @ np.array(
    [[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])


class BaseLineSearchFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def evaluate_grad(self, x):
        pass

    @abstractmethod
    def evaluate_hess(self, x):
        pass


class QuadraticFunction(BaseLineSearchFunction):
    def __init__(self, Q):
        self.Q = Q

    def evaluate(self, x):
        return x.transpose() @ self.Q @ x

    def evaluate_grad(self, x):
        return 2 * self.Q @ x

    def evaluate_hess(self, x=None):
        return 2 * self.Q


class RosenbrockFunction(BaseLineSearchFunction):
    def __init__(self):
        pass

    def evaluate(self, x):
        return 100 * np.power(x[1] - np.power(x[0], 2), 2) + np.power(1 - x[0], 2)

    def evaluate_grad(self, x):
        return np.array([[400 * np.power(x[0], 3) - 400 * x[0] * x[1] + 2 * x[0] - 2],
                         [-200 * np.power(x[0], 2) + 200 * x[1]]]).reshape((2, 1))

    def evaluate_hess(self, x):
        return np.array([[(1200 * np.power(x[0], 2) - 400 * x[1] + 2), (-400 * x[0])],
                         [(-400 * x[0]), np.array([200])]]).reshape(2, 2)


## constrined part ##

class QP_f0(BaseLineSearchFunction):
    def __init__(self, t=1.0):
        self.t = t

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    ## f0 = x0^2+x1^1+(x2+1)^2 ##
    def evaluate(self, x):
        x_copy = x.copy()
        x_copy[2] += 1
        return (x_copy ** 2).sum() * self.t

    def evaluate_grad(self, x):
        x_copy = x.copy()
        x_copy[2] += 1
        return 2 * x_copy * self.t

    def evaluate_hess(self, x=None):
        return 2 * np.eye(3) * self.t


class QP_f1(BaseLineSearchFunction):
    def __init__(self, t=1.0):
        pass

    ## f1 = x0
    def evaluate(self, x):
        return np.log(x[0])

    def evaluate_grad(self, x):
        return np.array([1 / x[0], np.array([0]), np.array([0])]).reshape(-1, 1)

    def evaluate_hess(self, x):
        hess = np.zeros((3, 3))
        hess[0, 0] = -1 / (x[0] ** 2)
        return hess


class QP_f2(BaseLineSearchFunction):
    def __init__(self, t=1.0):
        pass

    ## f2 = x1
    def evaluate(self, x):
        return np.log(x[1])

    def evaluate_grad(self, x):
        return np.array([np.array([0]), 1 / x[1], np.array([0])]).reshape(-1, 1)

    def evaluate_hess(self, x):
        hess = np.zeros((3, 3))
        hess[1, 1] = -1 / (x[1] ** 2)
        return hess


class QP_f3(BaseLineSearchFunction):
    def __init__(self, t=1.0):
        pass

    ## f3 = x2
    def evaluate(self, x):
        return np.log(x[2])

    def evaluate_grad(self, x):
        return np.array([np.array([0]), np.array([0]), 1 / x[2]]).reshape(-1, 1)

    def evaluate_hess(self, x):
        hess = np.zeros((3, 3))
        hess[2, 2] = -1 / (x[2] ** 2)
        return hess


class ConstrainedQuadraticFunction(BaseLineSearchFunction):
    def __init__(self):
        self.f0 = QP_f0()
        self.f1 = QP_f1()
        self.f2 = QP_f2()
        self.f3 = QP_f3()

    ## x0+x1+x2=1 ##
    @property
    def A(self):
        return np.array([1, 1, 1]).reshape(1, -1)

    @property
    def b(self):
        return np.array([1])

    @property
    def inequality_constraints(self):
        return [self.f1, self.f2, self.f3]

    def evaluate(self, x):
        ret_val = self.f0.evaluate(x)
        for fi in self.inequality_constraints:
            ret_val -= fi.evaluate(x)

        return ret_val

    def evaluate_grad(self, x):
        ret_val = self.f0.evaluate_grad(x)
        for fi in self.inequality_constraints:
            ret_val -= fi.evaluate_grad(x)

        return ret_val

    def evaluate_hess(self, x):
        ret_val = self.f0.evaluate_hess(x)
        for fi in self.inequality_constraints:
            ret_val -= fi.evaluate_hess(x)

        return ret_val

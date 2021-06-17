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

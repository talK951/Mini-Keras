import numpy as np


class ActivationFunction:

    def __call__(self, x: float):
        raise Exception("Activation Function is set to interface")

    def calc_derivative(self, x: float):
        pass


class Linear(ActivationFunction):

    def __call__(self, x: float):
        return x

    def calc_derivative(self, x: float):
        return 1


class Sigmoid(ActivationFunction):

    def __call__(self, x: float):
        return 1 / (1 + np.exp(-x))

    def calc_derivative(self, x: float):
        sigmoid_fn = Sigmoid()
        return sigmoid_fn(x) * (1 - sigmoid_fn(x))


class Tanh(ActivationFunction):

    def __call__(self, x: float):
        return np.tanh(x)

    def calc_derivative(self, x: float):
        tanh_fn = Tanh()
        return 1 - (tanh_fn(x) ** 2)


class Relu(ActivationFunction):

    def __call__(self, x):
        return max(x, 0)

    def calc_derivative(self, x: float):
        return 0 if x <= 0 else 1

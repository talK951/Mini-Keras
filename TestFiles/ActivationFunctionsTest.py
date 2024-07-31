import unittest
import numpy as np
from ActivationFunctions import Linear, Sigmoid, Relu, Tanh, ActivationFunction

input_values = [-2, -1, 0, 1, 2]


class TestActivationFunctions(unittest.TestCase):

    def test_activation_fn(self):
        activation_fn = ActivationFunction()
        for x in input_values:
            with self.assertRaises(Exception):
                activation_fn(x)

    def test_activation_fn_properties(self):
        assert issubclass(Linear, ActivationFunction)
        assert issubclass(Sigmoid, ActivationFunction)
        assert issubclass(Relu, ActivationFunction)
        assert issubclass(Tanh, ActivationFunction)

    def test_linear_fn(self):
        activation_fn = Linear()
        for x in input_values:
            assert activation_fn(x) == x

    def test_sigmoid_fn(self):
        activation_fn = Sigmoid()
        for x in input_values:
            assert activation_fn(x) == 1/(1 + np.exp(-x))

    def test_relu_fn(self):
        activation_fn = Relu()
        for x in input_values:
            assert activation_fn(x) == max(0, x)

    def test_tanh_fn(self):
        activation_fn = Tanh()
        for x in input_values:
            assert activation_fn(x) == np.tanh(x)

    def test_linear_derivative(self):
        activation_fn = Linear()
        assert activation_fn.calc_derivative(5) == 1
        assert activation_fn.calc_derivative(0) == 1
        assert activation_fn.calc_derivative(-5) == 1

    def test_sigmoid_derivative(self):
        activation_fn = Sigmoid()
        assert activation_fn.calc_derivative(0) == 0.25
        assert activation_fn.calc_derivative(1) < 0.25
        assert activation_fn.calc_derivative(-1) < 0.25

    def test_relu_derivative(self):
        activation_fn = Relu()
        assert activation_fn.calc_derivative(0) == 0
        assert activation_fn.calc_derivative(1) == 1
        assert activation_fn.calc_derivative(-1) == 0

    def test_tanh_derivative(self):
        activation_fn = Tanh()
        assert activation_fn.calc_derivative(0) == 1
        assert activation_fn.calc_derivative(1) < 0.5
        assert activation_fn.calc_derivative(-1) < 0.5

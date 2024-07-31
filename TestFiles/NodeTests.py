from Node import Node
import numpy as np
import unittest

input_values = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
input_values_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
input_size = len(input_values)


class TestNodeActivationFunctions(unittest.TestCase):
    assert input_size == len(weights)

    def test_relu_fn(self):
        node = Node(input_size, activation_function="relu")
        node._weights = weights
        assert node(input_values) == 0
        node._bias = 1
        assert node(input_values_2) == 10

    def test_sigmoid_fn(self):
        node = Node(input_size, activation_function="sigmoid")
        node._weights = weights
        assert node(input_values) == 0.5

    def test_tanh_fn(self):
        node = Node(input_size, activation_function="tanh")
        node._weights = weights
        assert node(input_values) == 0

    def test_linear_fn(self):
        node = Node(input_size, activation_function="linear")
        node._weights = weights
        assert node(input_values) == np.sum(input_values)
        node._bias = 1
        assert node(input_values_2) == 10

    def test_bias(self):
        node = Node(0, activation_function="linear")
        node._weights = []
        node._bias = 1
        assert node([]) == 1

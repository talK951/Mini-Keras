import numpy as np
from typing import Callable
from ActivationFunctions import Linear, Relu, Sigmoid, Tanh, ActivationFunction


class Node:
    _input_size: int
    _activation_function: ActivationFunction
    _weights: np.array
    _bias: float

    def __init__(self, input_size: int, activation_function: str) -> None:
        self._input_size = input_size
        self._set_activation_fn(activation_function)
        self._weights = np.zeros(input_size)
        self._bias = 0.0

    def __call__(self, input_values: np.array(float)) -> float:
        dot_product = np.dot(self._weights, input_values)
        return self._activation_function(dot_product + self._bias)

    def _set_activation_fn(self, activation_fn_name: str) -> None:
        if activation_fn_name == "linear":
            self._activation_function = Linear()
        elif activation_fn_name == "sigmoid":
            self._activation_function = Sigmoid()
        elif activation_fn_name == "tanh":
            self._activation_function = Tanh()
        elif activation_fn_name == "relu":
            self._activation_function = Relu()

    def partial_derivative(self, : np.array) -> np.array:
        for
        return self._activation_function(np.dot(x, self._weights))

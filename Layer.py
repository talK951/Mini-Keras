import numpy as np

from Node import Node
from typing import List


class Layer:
    nodes: List[Node]
    size: int

    def __init__(self, size: int, activation_fn: str) -> None:
        self.nodes = [Node(size, activation_function=activation_fn) for i in
                      range(size)]
        self.size = size

    def __call__(self, input_values: np.array(float)) -> np.array(float):
        pass


class Sequental(Layer):

    def __init__(self, size: int, activation_fn: str) -> None:
        Layer.__init__(self, size, activation_fn)

    def __call__(self, input_values: np.array(float)) -> np.array(float):
        nodes_output = [node(input_values) for node in self.nodes]
        return np.array(nodes_output)

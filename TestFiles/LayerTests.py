from Node import Node
from Layer import Sequental, Layer
import numpy as np
import unittest


class TestLayersFunctionality(unittest.TestCase):

    def test_layer_properties(self):
        assert issubclass(Sequental, Layer)

    def test_layer_propagation(self):
        input_layer = np.array([1, 1, 1, 1])
        size = len(input_layer)
        layer = Sequental(size, "relu")
        assert (layer(input_layer) == 0).all()

        input_layer = np.array([4, 4, 4, 4])
        size = len(input_layer)
        layer = Sequental(size, "relu")
        node = Node(1, "relu")
        node._weights = [1, 1, 1, 1]
        layer.nodes = [node, node, node, node]
        assert (layer(input_layer) == 16).all()

from Node import Node
from Layer import Sequental
from NueralNetwork import NueralNetwork
import numpy as np
import unittest


class TestLayersFunctionality(unittest.TestCase):

    def test_network_propagation(self):
        input_value = np.array([4, 4, 4, 4])
        size = len(input_value)
        node = Node(1, "relu")
        node._weights = [1, 1, 1, 1]
        node._bias = 1
        layer1 = Sequental(size, "relu")
        layer2 = Sequental(1, "relu")
        layer1.nodes = [node, node, node, node]
        layer2.nodes = [node]

        network = NueralNetwork(layer1, layer2)
        assert network(input_value) == 69

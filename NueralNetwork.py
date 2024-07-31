import numpy as np

import ErrorFunctions
from Layer import Layer
from typing import Union, List


class NueralNetwork:
    layers: List[Layer]
    _loss_fn: ErrorFunctions

    def __init__(self, *args: Union[Layer, BaseException]) -> None:
        self.layers = self._fetch_layers_from_args(*args)

    def _fetch_layers_from_args(self, *args: Union[Layer, BaseException]) -> List[Layer]:
        layers = []
        for obj in args:
            if not isinstance(obj, BaseException):
                layers.append(obj)
        return layers

    def __call__(self, input_values: np.array(float)) -> np.array([float]):
        prediction = input_values
        for layer in self.layers:
            prediction = layer(prediction)
        return prediction

    def compile(self, loss: ErrorFunctions) -> None:
        self._loss_fn = loss

    def fit(self, x_train: np.array, y_train: np.array, epochs: int) -> None:
        # Propagate into loss function

        # Begin backpropagation per epoch

        pass

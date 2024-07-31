import numpy as np
from typing import Optional


class ErrorFunction:
    def __call__(self, prediction: np.array(float),
                 ground_truth: np.array(float)):
        raise Exception("Error Function is set to interface")

    def check_input(self, prediction: np.array(float),
                    ground_truth: np.array(float)) -> Optional[Exception]:
        if len(prediction) != len(ground_truth):
            raise Exception("prediction or ground truth is diffent size")
        if len(prediction) == 0 or len(ground_truth) == 0:
            raise Exception("prediction or ground truth empty")

    def derivative(self, x: np.array, y: np.array):
        pass


class MSE(ErrorFunction):

    def __call__(self, prediction: np.array(float),
                 ground_truth: np.array(float)):
        self.check_input(prediction, ground_truth)
        return np.sum((prediction - ground_truth) ** 2) / len(prediction)

    def derivative(self, x: np.array, y: np.array):
        pass


class MAE(ErrorFunction):

    def __call__(self, prediction: np.array(float),
                 ground_truth: np.array(float)):
        self.check_input(prediction, ground_truth)
        return np.sum(np.absolute(prediction - ground_truth)) / len(prediction)


class LogLossError(ErrorFunction):

    def __call__(self, prediction: np.array(float),
                 ground_truth: np.array(float)):
        n = len(prediction)
        error = 0
        y_p = prediction
        y = ground_truth
        for i in range(n):
            if y_p[i] == 0 or y[i] == 0 or y_p[i] == 1 or y[i] == 1:
                error += 0
            else:
                error += y[i] * np.log(y_p) + (1 - y) * np.log(1 - y_p[i])
        return 0 if n == 0 else error / n


class RMSE(ErrorFunction):

    def __call__(self, prediction: np.array(float),
                 ground_truth: np.array(float)):
        self.check_input(prediction, ground_truth)
        number_of_data_points = len(prediction)
        error = np.sqrt((1 / number_of_data_points) * np.sum(
            (prediction - ground_truth) ** 2))
        return error

import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred += 1e-7
        n = len(y_pred)
        loss = -1 * (1 / n) * sum([(y_true[i] * np.log(y_pred[i])) + ((1 - y_true[i]) * np.log(1 - y_pred[i])) for i in range(n)]);

        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n_samples, n_classes = y_true.shape[0], y_true.shape[1]
        y_pred += 1e-7
        loss = 0

        for i in range(n_samples):
            for j in range(n_classes):
                loss += y_true[i][j] * np.log(y_pred[i][j])
        
        loss = -1 * (1 / n_samples) * loss

        return round(loss, 4)

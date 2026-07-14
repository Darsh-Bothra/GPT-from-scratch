import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def get_loss(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred += 1e-7
        n = len(y_pred)
        loss = -1 * (1 / n) * sum([(y_true[i] * np.log(y_pred[i])) + ((1 - y_true[i]) * np.log(1 - y_pred[i])) for i in range(n)]);

        return round(loss, 4)

        for i in range(n_samples):
            for j in range(n_classes):
                loss += y_true[i][j] * np.log(y_pred[i][j])
        
        loss = -1 * (1 / n_samples) * loss

        return round(loss, 4)

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        n, m = X.shape[0], X.shape[1]
        w = initial_weights
        for i in range(num_iterations):
            y_pred = self.get_model_prediction(X, w)

            for i in range(m):
                loss = self.get_loss(Y, y_pred)
                delta_loss = self.get_derivative(y_pred, Y, n, X, i)
                w[i] -= self.learning_rate * delta_loss
        
        return np.round(w, 5)









            

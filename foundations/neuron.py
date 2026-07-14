import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        y_pred = np.dot(x, w.T) + b
        if activation == "sigmoid":
            return round(1.0 / (1.0 + np.exp(-1 * y_pred)), 5)
        
        return round(float(max(0, y_pred)), 5)

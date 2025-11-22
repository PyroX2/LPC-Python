import numpy as np


class ARModel:
    def __init__(self, order: int = 4) -> None:
        self.order = order
        self.theta = np.zeros((order,1), dtype=np.float64)

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        self.theta = coeffs

    def _predict(self, x: np.ndarray) -> float:
        return float(self.theta.T @ x)
import numpy as np
import numpy.typing as npt

class ExactModelStep:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (input > self.threshold).astype(np.float32)
    
class ExactModelXSquared:
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sum(input ** 2, axis=1, keepdims=True)
    
class ExactModelXOR:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.logical_xor(input[:, 0] > self.threshold, input[:, 1] > self.threshold).astype(np.float32).reshape(-1, 1)
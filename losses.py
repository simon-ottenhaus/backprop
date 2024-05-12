from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> np.float32:
        raise NotImplementedError

    @abstractmethod
    def backward(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError
    
class MeanSquaredErrorLoss(Loss):
    def forward(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> np.float32:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    
    def forward_no_mean(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.y_pred = y_pred
        self.y_true = y_true
        return (y_pred - y_true) ** 2

    def backward(self) -> npt.NDArray[np.float32]:
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)
    

class CrossEntropyLoss(Loss):
    EPS = 1e-6

    def forward(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> np.float32:
        loss = self.forward_no_mean(y_pred, y_true)
        return np.mean(loss)
    
    def forward_no_mean(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Clipping prediction values to avoid log(0) scenario
        y_pred = np.clip(y_pred, self.EPS, 1 - self.EPS)
        self.y_pred = y_pred
        self.y_true = y_true
        # Calculate the negative log likelihood
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return loss

    def backward(self) -> npt.NDArray[np.float32]:
        # Calculating the gradient
        self.y_pred = np.clip(self.y_pred, self.EPS, 1 - self.EPS)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred)) / len(self.y_true)
        return grad
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from tqdm import trange
import matplotlib.pyplot as plt

class Layer(ABC):
    @abstractmethod
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Forward pass of the layer. It should return the output of the layer.

        Computes the output of the layer given the input. It should also store the input and output
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    def update_parameters(self, learning_rate: float) -> None:
        pass  # Some layers might not have parameters

class Sigmoid(Layer):
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        sigmoid_derivative = self.output * (1 - self.output)
        return output_gradient * sigmoid_derivative

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((output_size))
    

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Y = XW + B

        X: input of shape (batch_size, input_size)
        W: weights of shape (input_size, output_size)
        B: biases of shape (output_size)
        Y: output of shape (batch_size, output_size)
        """
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Y = XW + B
        dY/dW = X
        dY/dX = W
        dY/dB = 1

        dW = X.T * dY
        dB = sum(dY)
        dX = dY * W.T
        """
        weights_gradient = np.dot(self.input.T, output_gradient)
        self.weights_gradient = weights_gradient
        self.biases_gradient = np.sum(output_gradient, axis=0)
        output_gradient = np.dot(output_gradient, self.weights.T)
        return output_gradient

    def update_parameters(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

class WeightLayer(Layer):
    """
    Y = XW
    """

    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * 0.1

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        return np.dot(input, self.weights)

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        return np.dot(output_gradient, self.weights.T)

    def update_parameters(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_gradient


class BiasLayer(Layer):
    """
    Y = X + B
    """

    def __init__(self, input_size: int, output_size: int):
        self.biases = np.zeros((output_size))

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        return input + self.biases

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.biases_gradient = np.sum(output_gradient, axis=0)
        return output_gradient

    def update_parameters(self, learning_rate: float) -> None:
        self.biases -= learning_rate * self.biases_gradient


class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> None:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_parameters(learning_rate)

class MeanSquaredErrorLoss:
    def forward(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> np.float32:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> npt.NDArray[np.float32]:
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)
    
class ExactModelXSquared:
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sum(input ** 2, axis=1, keepdims=True)
    
class ExactModelXOR:
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.logical_xor(input[:, 0] > 0.5, input[:, 1] > 0.5).astype(np.float32).reshape(-1, 1)

def plot_3d(X: npt.NDArray[np.float32], y: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, label='True')
    ax.scatter(X[:, 0], X[:, 1], y_pred, label='Predicted')
    plt.legend()
    plt.show()

def train_x_squared():
    # Example usage
    layers = [
        Dense(2, 4),
        Sigmoid(),
        Dense(4, 1),
        Sigmoid(),
        Dense(1, 1),
    ]
    model = NeuralNetwork(layers)
    loss = MeanSquaredErrorLoss()

    exact_model = ExactModelXSquared()

    # # XOR dataset
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    # y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Y = X1^2 + X2^2
    # X = 0..1 in 0.1 increments
    X = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    print(X.shape)
    y = exact_model.forward(X)
    print(y.shape)

    # track the loss
    loss_values: list[float] = []

    for epoch in trange(100_000):
        output = model.forward(X)
        loss_value = loss.forward(output, y)
        loss_values.append(loss_value.item())

        loss_gradient = loss.backward()
        model.backward(loss_gradient)
        model.update_parameters(0.1)

        if epoch % 10_000 == 0:
            print(f"Epoch {epoch}, loss {loss_value}")
            print("Predictions")
            # print(model.forward(X))

    pred_y = model.forward(X)
    # make a plot
    plot_3d(X, y, pred_y)

def train_xor():
    layers = [
        Dense(2, 16),
        Sigmoid(),
        Dense(16, 16),
        Sigmoid(),
        Dense(16, 1),
        Sigmoid(),
    ]
    model = NeuralNetwork(layers)
    loss = MeanSquaredErrorLoss()

    exact_model = ExactModelXOR()

    # XOR dataset
    X = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    y = exact_model.forward(X)

    pbar = trange(100_000)

    for epoch in pbar:
        output = model.forward(X)
        loss_value = loss.forward(output, y)
        loss_gradient = loss.backward()
        model.backward(loss_gradient)
        model.update_parameters(0.1)

        if epoch % 10_000 == 0:
            pbar.set_description(f"Epoch {epoch}, loss {loss_value:.4f}")

    pred_y = model.forward(X)
    # make a plot
    plot_3d(X, y, pred_y)

class ExactModelStep:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (input > self.threshold).astype(np.float32)
    
def train_super_simple():
    # 1 Layer, 1 Neuron, 1 Input
    dense = Dense(1, 1)
    # set Weight = 1, Bias = 0
    dense.weights = np.ones_like(dense.weights)
    dense.biases = np.zeros_like(dense.biases)

    layers = [
        dense,
        Sigmoid(),
    ]
    model = NeuralNetwork(layers)
    loss = MeanSquaredErrorLoss()

    exact_model = ExactModelStep(2)

    X = np.linspace(0, 4, 11).reshape(-1, 1)
    y = exact_model.forward(X)
    print(f"X: {X}")
    print(f"y: {y}")

    pbar = trange(100_000)

    LR = 0.5

    for epoch in pbar:
        output = model.forward(X)
        loss_value = loss.forward(output, y)
        loss_gradient = loss.backward()
        model.backward(loss_gradient)
        model.update_parameters(LR)

        if epoch % 10_000 == 0:
            pbar.set_description(f"Epoch {epoch}, loss {loss_value:.4f}")
            # print Weight and Bias
            print(f"Weight: {dense.weights}")
            print(f"Bias: {dense.biases}")

    pred_y = model.forward(X)
    # make a plot
    plt.plot(X, y, label='True')
    plt.plot(X, pred_y, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # train_x_squared()
    # train_xor()
    train_super_simple()

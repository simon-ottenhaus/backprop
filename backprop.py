from abc import ABC, abstractmethod
from typing import Type

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

    @abstractmethod
    def clone(self) -> "Layer":
        raise NotImplementedError


class Sigmoid(Layer):
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # sigmoid(x) = 1 / (1 + exp(-x))
        # sigmoid(0) = 0.5
        # sigmoid(-inf) = 0
        # sigmoid(+inf) = 1
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # d(sigmoid(x))/dx = d/dx(1 / (1 + exp(-x))) = exp(-x) / (1 + exp(-x))^2 = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = self.output * (1 - self.output)
        return output_gradient * sigmoid_derivative
    
    def clone(self) -> "Layer":
        return Sigmoid()
    
class ReLU(Layer):
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return output_gradient * (self.input > 0)
    
    def clone(self) -> "Layer":
        return ReLU()
    
class Tanh(Layer):
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # tanh(0) = 0
        # tanh(-inf) = -1
        # tanh(+inf) = 1
        self.input = input
        return np.tanh(input)
    
    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # d(tanh(x))/dx = d/dx((exp(x) - exp(-x)) / (exp(x) + exp(-x))) = 1 - tanh(x)^2
        return output_gradient * (1 - np.tanh(self.input) ** 2)
    
    def clone(self) -> "Layer":
        return Tanh()
    
class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.input = input
        return np.maximum(self.alpha * input, input)
    
    def backward(self, output_gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return output_gradient * ((self.input > 0) + self.alpha * (self.input <= 0))
    
    def clone(self) -> "Layer":
        return LeakyReLU(self.alpha)

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, initializer: str = "normal"):
        if initializer == "normal":
            self.weights, self.biases = self.initialize_normal(input_size, output_size)
        elif initializer == "deterministic_linear":
            self.weights, self.biases = self.initialize_deterministic_linear(input_size, output_size)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
        
    def clone(self) -> "Layer":
        new_layer = Dense(self.weights.shape[0], self.weights.shape[1])
        new_layer.weights = self.weights.copy()
        new_layer.biases = self.biases.copy()
        return new_layer


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

    def initialize_deterministic_linear(self, input_size: int, output_size: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        weights = np.linspace(0, 1, num=input_size * output_size).reshape(input_size, output_size)
        biases = np.linspace(0, 1, num=output_size)
        return weights, biases
    
    def initialize_normal(self, input_size: int, output_size: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        weights = np.random.randn(input_size, output_size) * 0.1
        biases = np.zeros_like(output_size)
        return weights, biases


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
    
    def forward_with_intermediates(self, input: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        intermediates: list[npt.NDArray[np.float32]] = []
        for layer in self.layers:
            input = layer.forward(input)
            intermediates.append(input)
        return intermediates

    def backward(self, output_gradient: npt.NDArray[np.float32]) -> None:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def backward_with_intermediates(self, output_gradient: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        intermediates: list[npt.NDArray[np.float32]] = []
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
            intermediates.append(output_gradient)
        return intermediates

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def clone(self) -> "NeuralNetwork":
        new_layers = [layer.clone() for layer in self.layers]
        return NeuralNetwork(new_layers)

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
    def forward(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> np.float32:
        loss = self.forward_no_mean(y_pred, y_true)
        return np.mean(loss)
    
    def forward_no_mean(self, y_pred: npt.NDArray[np.float32], y_true: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Clipping prediction values to avoid log(0) scenario
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        self.y_pred = y_pred
        self.y_true = y_true
        # Calculate the negative log likelihood
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return loss

    def backward(self) -> npt.NDArray[np.float32]:
        # Calculating the gradient
        eps = 1e-15
        self.y_pred = np.clip(self.y_pred, eps, 1 - eps)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred)) / len(self.y_true)
        return grad
    
class ExactModelXSquared:
    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.sum(input ** 2, axis=1, keepdims=True)
    
class ExactModelXOR:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def forward(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.logical_xor(input[:, 0] > self.threshold, input[:, 1] > self.threshold).astype(np.float32).reshape(-1, 1)

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

    exact_model = ExactModelXOR(0.5)

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


def train_minimal_xor(
        hiden_size: int = 4, 
        train_resolution: int = 10,
        inner_activation: Type[Layer] = Sigmoid,
        ):
    """
    Train two layer neural network on the minimal XOR dataset. Visualize the hidden layer outputs over time.
    The network has 2 input neurons, a hidden layer with `hiden_size` neurons, and an output layer with 1 neuron.
    
    Args:
        hiden_size: Number of neurons in the hidden layer. Should be at least 4 to learn XOR.
        train_resolution: Number of points in each dimension of the training grid. The total number of training points is `train_resolution`^2.
        activation: Activation function to use in the hidden layer. Should be a class that inherits from `Layer`. E.g. `Sigmoid`, `ReLU`, `Tanh`.
    """
    
    # use a deterministic initializer to always get the same results
    layers = [
        Dense(2, hiden_size, initializer="deterministic_linear"),
        inner_activation(),
        Dense(hiden_size, 1, initializer="deterministic_linear"),
        Sigmoid(),
    ]

    model = NeuralNetwork(layers)
    # loss = MeanSquaredErrorLoss()
    loss = CrossEntropyLoss()

    exact_model = ExactModelXOR(0.5)

    # Minimal XOR dataset
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    X = np.meshgrid(np.linspace(0, 1, train_resolution), np.linspace(0, 1, train_resolution))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    y = exact_model.forward(X)

    pbar = trange(30_000)
    intermediate_models: list[NeuralNetwork] = []
    LR = 0.5

    for epoch in pbar:
        output = model.forward(X)
        loss_value = loss.forward(output, y)
        loss_gradient = loss.backward()
        model.backward(loss_gradient)
        model.update_parameters(LR)

        if epoch % 2_000 == 0:
            pbar.set_description(f"Epoch {epoch}, loss {loss_value:.4f}")
            intermediate_models.append(model.clone())

    # X_test = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    # X_test = np.stack(X_test, axis=-1).reshape(-1, 2)
    # pred_y = model.forward(X_test)
    # y_test = exact_model.forward(X_test)
    # # make a plot
    # plot_3d(X_test, y_test, pred_y)

    # visualize the layer outputs over time
    X_grid = np.meshgrid(np.linspace(0, 1, 51), np.linspace(0, 1, 51))
    X_test = np.stack(X_grid, axis=-1).reshape(-1, 2)
    Y_test = exact_model.forward(X_test)

    fig, axs = plt.subplots(hiden_size + 2 + len(model.layers) + 1, len(intermediate_models), figsize=(15, 5))

    for i, model in enumerate(intermediate_models):
        intermediates = model.forward_with_intermediates(X_test)

        out_1 = intermediates[0]
        out_2 = intermediates[2]
        
        # use red white blue colormap
        for j in range(hiden_size):
            out_grid_1 = out_1[:, j].reshape(51, 51)
            axs[j, i].imshow(out_grid_1, extent=(0, 1, 0, 1), vmin=-1, vmax=1, cmap='viridis')

        out_grid_2 = out_2[:, 0].reshape(51, 51)
        axs[hiden_size + 0, i].imshow(out_grid_2, extent=(0, 1, 0, 1), vmin=-1, vmax=1, cmap='viridis')

        # add loss to the plot
        loss_values = loss.forward_no_mean(model.forward(X_test), Y_test)
        loss_grid = loss_values.reshape(51, 51)
        axs[hiden_size + 1, i].imshow(loss_grid, extent=(0, 1, 0, 1), vmin=0, vmax=1)

        loss_gradient = loss.backward()
        gradients = [loss_gradient]
        gradients += model.backward_with_intermediates(loss_gradient)
        gradients = gradients[::-1]
        # print(len(gradients))
        gradients_absmax = max([np.abs(g).max() for g in gradients])

        for j, gradient in enumerate(gradients):
            out_grid = gradient[:, 0].reshape(51, 51)
            axs[hiden_size + 2 + j, i].imshow(out_grid, extent=(0, 1, 0, 1), vmin=-gradients_absmax, vmax=gradients_absmax, cmap='RdBu')

        # remove axis labels and ticks
        for ax in axs[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()



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
    # train_super_simple()
    train_minimal_xor()

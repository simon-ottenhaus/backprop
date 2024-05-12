import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

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
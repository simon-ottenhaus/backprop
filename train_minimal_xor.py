
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from exact_models import ExactModelXOR
from layers import Dense, Layer, NeuralNetwork, Sigmoid
from losses import CrossEntropyLoss
from plotting import plot_3d


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


    X_test = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    X_test = np.stack(X_test, axis=-1).reshape(-1, 2)
    pred_y = model.forward(X_test)
    y_test = exact_model.forward(X_test)
    # make a plot
    plot_3d(X_test, y_test, pred_y)


if __name__ == "__main__":
    train_minimal_xor()

import numpy as np
from tqdm import trange

from exact_models import ExactModelXOR
from layers import Dense, NeuralNetwork, Sigmoid
from losses import MeanSquaredErrorLoss
from plotting import plot_3d


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


if __name__ == "__main__":
    train_xor()
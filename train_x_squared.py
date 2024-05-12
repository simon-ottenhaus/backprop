import numpy as np
from tqdm import trange

from exact_models import ExactModelXSquared
from layers import Dense, NeuralNetwork, Sigmoid
from losses import MeanSquaredErrorLoss
from plotting import plot_3d


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


if __name__ == "__main__":
    train_x_squared()
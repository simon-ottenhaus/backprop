import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from exact_models import ExactModelStep
from layers import Dense, NeuralNetwork, Sigmoid
from losses import MeanSquaredErrorLoss


def train_step_function():
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
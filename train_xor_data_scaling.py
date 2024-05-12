from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import trange

from exact_models import ExactModelXOR
from layers import Dense, NeuralNetwork, Sigmoid, Tanh
from losses import CrossEntropyLoss


def accuracy(y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> float:
    pred_classes = y_pred > 0.5
    true_classes = y_true > 0.5
    return np.mean(pred_classes == true_classes)


class TrainResult(NamedTuple):
    model: NeuralNetwork
    error_rate: float

def train_xor_data_scaling(n_train: int) -> TrainResult:
    hidden_dim = 16
    epochs = 50_000
    learning_rate = 0.1


    layers = [
        Dense(2, hidden_dim),
        Tanh(),
        Dense(hidden_dim, hidden_dim),
        Tanh(),
        Dense(hidden_dim, 1),
        Sigmoid(),
    ]
    model = NeuralNetwork(layers)
    loss = CrossEntropyLoss()
    exact_model = ExactModelXOR(0.5)

    X = np.random.uniform(0, 1, (n_train, 2)).astype(np.float32)
    y = exact_model.forward(X)

    pbar = trange(epochs)

    for epoch in pbar:
        output = model.forward(X)
        loss_value = loss.forward(output, y)
        loss_gradient = loss.backward()
        model.backward(loss_gradient)
        model.update_parameters(learning_rate)

        if epoch % 1_000 == 0:
            pbar.set_description(f"Epoch {epoch}, loss {loss_value:.4f}")

    X_test = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X_test = np.stack(X_test, axis=-1).reshape(-1, 2)
    pred_y = model.forward(X_test)
    y_test = exact_model.forward(X_test)

    acc = accuracy(y_test, pred_y)
    error_rate = 1 - acc
    # print(f"Accuracy: {acc:.2f}")
    print(f"Error rate: {error_rate:.2%}")
    
    return TrainResult(model, error_rate)



if __name__ == "__main__":
    n_runs = 10
    train_amounts = [10, 20, 50, 100, 200, 500, 1000]
    error_rates = np.zeros((n_runs, len(train_amounts)), dtype=np.float32)
    records: list[dict] = []
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        for j, n_train in enumerate(train_amounts):
            result = train_xor_data_scaling(n_train)
            error_rates[i, j] = result.error_rate
            records.append({"run": i+1, "n_train": n_train, "error_rate": result.error_rate})

    df = pd.DataFrame(records)
    df.to_csv("xor_data_scaling.csv", index=False)

    # for each run plot the error over the number of training samples, linewidth=0.5
    for i in range(n_runs):
        plt.plot(train_amounts, error_rates[i], label=f"Run {i+1}", color="gray", linewidth=0.5)

    # plot the median error over the number of training samples
    avg_error_rates = np.median(error_rates, axis=0)
    plt.plot(train_amounts, avg_error_rates, label="Median", color="black", linewidth=2)

    plt.xscale("log")
    plt.xlabel("Number of training samples")
    plt.yscale("log")
    plt.ylabel("Error rate")
    plt.legend()
    plt.savefig("xor_data_scaling.png")
    plt.show()


    
    

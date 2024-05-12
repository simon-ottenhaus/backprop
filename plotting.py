import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_3d(X: npt.NDArray[np.float32], y: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, label='True')
    ax.scatter(X[:, 0], X[:, 1], y_pred, label='Predicted')
    plt.legend()
    plt.show()
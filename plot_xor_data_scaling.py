import pandas as pd
import matplotlib.pyplot as plt

def plot_xor_data_scaling():
    df = pd.read_csv("xor_data_scaling.csv")
    # plot the error rate for each run
    df.pivot(index="n_train", columns="run", values="error_rate").plot(legend=False, color='gray', linewidth=0.5)
    # plot the median error rate and mark the data points
    df.groupby("n_train")["error_rate"].median().plot(label="Median", color='black', linewidth=2, marker='o', markersize=5)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of training samples")
    plt.ylabel("Test error rate")
    plt.legend()
    plt.title("Log-Log Linearity of Data Scaling")
    plt.tight_layout()
    plt.savefig("xor_data_scaling.png")
    # plt.show()



if __name__ == "__main__":
    plot_xor_data_scaling()
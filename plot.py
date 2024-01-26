import numpy as np
import matplotlib.pyplot as plt

# Feel free to change the figure size and apply other improvements to these basic functions.
# Use plt.save_fig instead of plt.show (or before that) to save your figures. 

def plot_mean_error(mean_array, title: str = ""):
    plt.figure(figsize=(6, 4))
    plt.plot(mean_array)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Mean Error')
    plt.show()

def plot_squared_sum_diag_cov(covariance_mat, title: str = ""):
    diag_sum = np.sum(np.square(np.diag(covariance_mat)))
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(covariance_array)), [diag_sum] * len(covariance_array))
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Covariance')
    plt.show()

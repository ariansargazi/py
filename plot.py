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

def plot_squared_sum_diag_cov(covariance_matrices, title: str = ""):
    plt.figure(figsize=(6, 4))

    diag_sums = []
    for covariance_mat in covariance_matrices:
        diag_sums.append(np.sum(np.square(np.diag(covariance_mat))))
    
    plt.plot(diag_sums)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Covariance')
    plt.show()

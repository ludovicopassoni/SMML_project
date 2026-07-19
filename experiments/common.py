from src.data import make_synthetic_classification_dataset, load_real_world_dataset
import matplotlib.pyplot as plt
from src.kernels import median_heuristic_gamma
import numpy as np

def load_all_datasets():
    return {
        "synthetic": make_synthetic_classification_dataset(),
        "real": load_real_world_dataset(),
    }

def save_and_show(dataset_name, plot_type):
    plt.savefig(f"plots/{dataset_name}_{plot_type}.png", dpi=150, bbox_inches="tight")
    plt.show()

def make_gamma_list(X_train, radius=1.5, num=10):
    gamma_heuristic = median_heuristic_gamma(X_train)
    log_center = np.log10(gamma_heuristic)
    return np.logspace(log_center - radius, log_center + radius, num=num)
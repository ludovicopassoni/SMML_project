import numpy as np
import matplotlib.pyplot as plt
from experiments.common import load_all_datasets, save_and_show, make_gamma_list
from src.svm_smo import KernelSVM_SMO
from src.kernels import median_heuristic_gamma

def run_gamma_experiment(X_train, X_test, y_train, y_test, gamma_list, dataset_name):
    train_accs = []
    test_accs = []

    for gamma in gamma_list:
        model = KernelSVM_SMO(gamma=gamma)
        model.fit(X_train, y_train)

        train_accs.append(model.score(X_train, y_train))
        test_accs.append(model.score(X_test, y_test))

    return train_accs, test_accs

def plot_gamma_effect(gamma_list, train_accs, test_accs, dataset_name):
    plt.figure()
    plt.plot(gamma_list, train_accs, marker="o", label="Train accuracy")
    plt.plot(gamma_list, test_accs, marker="o", label="Test accuracy")
    plt.xscale("log")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.title(f"Effect of RBF bandwidth (gamma) on {dataset_name} dataset")
    plt.legend()
    plt.grid()
    save_and_show(dataset_name, "gamma_effect")

if __name__ == "__main__":
    datasets = load_all_datasets()

    X_train, X_test, y_train, y_test = datasets["synthetic"]
    gamma_list = make_gamma_list(X_train)
    train_accs, test_accs = run_gamma_experiment(X_train, X_test, y_train, y_test, gamma_list, "synthetic")
    plot_gamma_effect(gamma_list, train_accs, test_accs, "synthetic")

    X_train_real, X_test_real, y_train_real, y_test_real = datasets["real"]
    gamma_list_real = make_gamma_list(X_train_real)
    train_accs_real, test_accs_real = run_gamma_experiment(X_train_real, X_test_real, y_train_real, y_test_real, gamma_list_real, "real")
    plot_gamma_effect(gamma_list_real, train_accs_real, test_accs_real, "real")
    print(f"[synthetic] min test acc={min(test_accs):.4f}, max test acc={max(test_accs):.4f}")
    print(f"[real] min test acc={min(test_accs_real):.4f}, max test acc={max(test_accs_real):.4f}")
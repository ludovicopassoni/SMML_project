import numpy as np
import matplotlib.pyplot as plt
from experiments.common import load_all_datasets, save_and_show
from src.linear_model import LinearSVM
from src.svm_smo import KernelSVM_SMO
from src.random_features import RandomFourierFeatures
from src.kernels import median_heuristic_gamma


def run_experiment(X_train, X_test, y_train, y_test, n_components_list, dataset_name):
    gamma = median_heuristic_gamma(X_train)

    linear_baseline = LinearSVM()
    linear_baseline.fit(X_train, y_train)
    baseline_test_acc = linear_baseline.score(X_test, y_test)

    kernel_svm = KernelSVM_SMO(gamma=gamma)
    kernel_svm.fit(X_train, y_train)
    kernel_test_acc = kernel_svm.score(X_test, y_test)

    train_accs = []
    test_accs = []

    for D in n_components_list:
        rff = RandomFourierFeatures(n_components=D, gamma=gamma, random_state=0)
        Z_train = rff.fit_transform(X_train)
        Z_test = rff.transform(X_test)

        model = LinearSVM()
        model.fit(Z_train, y_train)

        train_accs.append(model.score(Z_train, y_train))
        test_accs.append(model.score(Z_test, y_test))

    return train_accs, test_accs, baseline_test_acc, kernel_test_acc

def plot_results(n_components_list, train_accs, test_accs, baseline_test_acc, kernel_test_acc, dataset_name):
    plt.figure()
    plt.plot(n_components_list, test_accs, marker="o", label="RFF + Linear (test)")
    plt.plot(n_components_list, train_accs, marker="o", label="RFF + Linear (train)")
    plt.axhline(y=baseline_test_acc, color='r', linestyle='--', label="Linear SVM (test)")
    plt.axhline(y=kernel_test_acc, color='g', linestyle='--', label="Kernel SVM (test)")
    plt.xscale('log')
    plt.xlabel("Number of Random Fourier Features (D)")
    plt.ylabel("Accuracy")
    plt.title(f"Performance on {dataset_name} dataset")
    plt.legend()
    plt.grid()
    save_and_show(dataset_name, "rff_comparison")

if __name__ == "__main__":
    n_components_list = [5, 10, 25, 50, 100, 250, 500, 1000]

    datasets = load_all_datasets()
    X_train_synth, X_test_synth, y_train_synth, y_test_synth = datasets["synthetic"]
    X_train_real, X_test_real, y_train_real, y_test_real = datasets["real"]
    results_synth = run_experiment(X_train_synth, X_test_synth, y_train_synth, y_test_synth, n_components_list, "synthetic")
    results_real = run_experiment(X_train_real, X_test_real, y_train_real, y_test_real, n_components_list, "real")
    plot_results(n_components_list, *results_synth, "synthetic")
    plot_results(n_components_list, *results_real, "real")
    print(f"[synthetic] baseline={results_synth[2]:.4f}, kernel={results_synth[3]:.4f}, best RFF={max(results_synth[1]):.4f}")
    print(f"[real] baseline={results_real[2]:.4f}, kernel={results_real[3]:.4f}, best RFF={max(results_real[1]):.4f}")
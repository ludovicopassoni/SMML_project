import time
import numpy as np
import matplotlib.pyplot as plt
from experiments.common import load_all_datasets, save_and_show
from src.linear_model import LinearSVM
from src.svm_smo import KernelSVM_SMO
from src.random_features import RandomFourierFeatures
from src.kernels import median_heuristic_gamma



def collect_runtime_accuracy_data(X_train, X_test, y_train, y_test, n_components_list, dataset_name):
    results = []  # lista di dizionari o tuple, uno per ogni configurazione testata
    t0 = time.time()
    linear_baseline = LinearSVM()
    linear_baseline.fit(X_train, y_train)
    baseline_runtime = time.time() - t0
    baseline_test_acc = linear_baseline.score(X_test, y_test)
    results.append(("Linear Baseline", baseline_runtime, baseline_test_acc))

    gamma = median_heuristic_gamma(X_train)
    t0 = time.time()
    kernel_svm = KernelSVM_SMO(gamma=gamma)
    kernel_svm.fit(X_train, y_train)
    kernel_runtime = time.time() - t0
    kernel_test_acc = kernel_svm.score(X_test, y_test)
    results.append(("Kernel SVM (exact)", kernel_runtime, kernel_test_acc))

    for D in n_components_list:
        t0 = time.time()
        rff = RandomFourierFeatures(n_components=D, gamma=gamma, random_state=0)
        Z_train = rff.fit_transform(X_train)
        Z_test = rff.transform(X_test)
        model = LinearSVM()
        model.fit(Z_train, y_train)
        rff_runtime = time.time() - t0
        rff_test_acc = model.score(Z_test, y_test)
        results.append((f"RFF (D={D})", rff_runtime, rff_test_acc))
    return results

def plot_runtime_accuracy(results, dataset_name):
    labels, times, accs = zip(*results)

    plt.figure()
    plt.scatter(times, accs)
    plt.xscale("log")
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Test accuracy")
    plt.title(f"Runtime vs Accuracy trade-off ({dataset_name})")
    plt.grid()

    for label, t, acc in zip(labels, times, accs):
        plt.annotate(label, (t, acc), textcoords="offset points", xytext=(0,10), ha='center')
    save_and_show(dataset_name, "runtime_accuracy")

if __name__ == "__main__":
    datasets = load_all_datasets()
    # Synthetic dataset
    X_train, X_test, y_train, y_test = datasets["synthetic"]
    n_components_list = [10, 50, 100, 200]
    results = collect_runtime_accuracy_data(X_train, X_test, y_train, y_test, n_components_list, "synthetic")
    plot_runtime_accuracy(results, "synthetic")
    for label, runtime, acc in results:
        print(f"[synthetic] {label}: time={runtime:.4f}s, test_acc={acc:.4f}")

    # Real-world dataset
    X_train, X_test, y_train, y_test = datasets["real"]
    n_components_list = [100, 500, 1000]
    results = collect_runtime_accuracy_data(X_train, X_test, y_train, y_test, n_components_list, "real")
    plot_runtime_accuracy(results, "real")
    for label, runtime, acc in results:
        print(f"[real] {label}: time={runtime:.4f}s, test_acc={acc:.4f}")
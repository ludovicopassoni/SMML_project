from src.polynomial_features import PolynomialFeatureMap
from src.linear_model import LinearSVM
from experiments.common import load_all_datasets

def evaluate_polynomial_features(X_train, X_test, y_train, y_test, degree):
    poly = PolynomialFeatureMap(degree=degree)
    Z_train = poly.fit_transform(X_train)
    Z_test = poly.transform(X_test)

    model = LinearSVM()
    model.fit(Z_train, y_train)

    train_acc = model.score(Z_train, y_train)
    test_acc = model.score(Z_test, y_test)

    return train_acc, test_acc, Z_train.shape[1]  # Return also the number of features generated

if __name__ == "__main__":
    datasets = load_all_datasets()
    degrees = [2, 3]

    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        for degree in degrees:
            train_acc, test_acc, n_features = evaluate_polynomial_features(
                X_train, X_test, y_train, y_test, degree
            )
            print(f"[{dataset_name}] degree={degree}: "
                  f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, n_features={n_features}")
import numpy as np


class LinearSVM:
    def __init__(self, lambda_reg=1e-3, n_epochs=200, random_state=0):
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        # Homogeneous coordinates trick: fold the bias into w by appending
        # a constant feature of 1s to every point. w_ then has dimension
        # d+1, and its last component acts as the (now regularized) bias.
        X_aug = np.hstack([X, np.ones((n, 1))])
        self.w_ = np.zeros(d + 1)

        t = 0  # global iteration counter, shared across all epochs
        for epoch in range(self.n_epochs):
            indices = rng.permutation(n)
            for i in indices:
                t += 1
                lr = 1.0 / (self.lambda_reg * t)  # Pegasos learning rate

                margin = y[i] * np.dot(X_aug[i], self.w_)
                if margin < 1:
                    grad = self.lambda_reg * self.w_ - y[i] * X_aug[i]
                else:
                    grad = self.lambda_reg * self.w_
                self.w_ -= lr * grad

        return self

    def decision_function(self, X):
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])
        return X_aug @ self.w_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        predictions = self.predict(X)
        predictions[predictions == 0] = 1.0
        return np.mean(predictions == y)
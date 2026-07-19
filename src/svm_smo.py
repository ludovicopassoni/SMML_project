import numpy as np
from src.kernels import rbf_kernel_matrix


class KernelSVM_SMO:
    def __init__(self, C=1.0, gamma=0.5, tol=1e-3, max_passes=10, random_state=0):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

    def fit(self, X, y):
        n, d = X.shape
        self.alpha_ = np.zeros(n)
        self.b_ = 0.0
        K = rbf_kernel_matrix(X, X, self.gamma)
        self.X_train_ = X
        self.y_train_ = y
        def f(i):
            return np.sum(self.alpha_ * self.y_train_ * K[:, i]) + self.b_
        rng = np.random.default_rng(self.random_state)
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n):
                E_i = f(i) - self.y_train_[i]
                if (self.y_train_[i] * E_i < -self.tol and self.alpha_[i] < self.C) or (self.y_train_[i] * E_i > self.tol and self.alpha_[i] > 0):
                    j = i
                    while j == i:
                        j = rng.integers(0, n)
                    E_j = f(j) - self.y_train_[j]
                    alpha_i_old = self.alpha_[i]
                    alpha_j_old = self.alpha_[j]
                    if self.y_train_[i] != self.y_train_[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    if L == H:
                        continue
                    eta = 2*K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    self.alpha_[j] = alpha_j_old - self.y_train_[j] * (E_i - E_j) / eta
                    self.alpha_[j] = min(H, max(L, self.alpha_[j]))
                    if abs(self.alpha_[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha_[i] = alpha_i_old + self.y_train_[i] * self.y_train_[j] * (alpha_j_old - self.alpha_[j])
                    b1 = self.b_ - E_i - self.y_train_[i] * (self.alpha_[i] - alpha_i_old) * K[i,i] - self.y_train_[j] * (self.alpha_[j] - alpha_j_old) * K[i,j]
                    b2 = self.b_ - E_j - self.y_train_[i] * (self.alpha_[i] - alpha_i_old) * K[i,j] - self.y_train_[j] * (self.alpha_[j] - alpha_j_old) * K[j,j]
                    if 0 < self.alpha_[i] < self.C:
                        self.b_ = b1
                    elif 0 < self.alpha_[j] < self.C:
                        self.b_ = b2
                    else:
                        self.b_ = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self

    def decision_function(self, X):
        K_test = rbf_kernel_matrix(X, self.X_train_, self.gamma)
        scores = K_test @ (self.alpha_ * self.y_train_) + self.b_
        return scores

    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
import numpy as np


class RandomFourierFeatures:
    def __init__(self, n_components, gamma, random_state=0):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        self.W_ = rng.normal(loc=0.0, scale=np.sqrt(2 * self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        projection = X @ self.W_ + self.b_
        Z = np.sqrt(2 / self.n_components) * np.cos(projection)
        return Z
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
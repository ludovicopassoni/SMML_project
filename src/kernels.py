import numpy as np

def squared_distances(X1, X2):
    sq_norms1 = np.sum(X1 ** 2, axis=1)
    sq_norms2 = np.sum(X2 ** 2, axis=1)
    cross_term = X1 @ X2.T

    sq_dists = sq_norms1.reshape(-1, 1) + sq_norms2.reshape(1, -1) - 2 * cross_term
    sq_dists = np.maximum(sq_dists, 0.0)
    return sq_dists

def rbf_kernel_matrix(X1, X2, gamma):
    sq_dists = squared_distances(X1, X2)
    K = np.exp(-gamma * sq_dists)
    return K

def median_heuristic_gamma(X):
    sq_dists = squared_distances(X, X)
    upper_tri_indices = np.triu_indices_from(sq_dists, k=1)
    upper_tri_sq_dists = sq_dists[upper_tri_indices]
    median_sq_dist = np.median(upper_tri_sq_dists)
    gamma = 1.0 / (2 * median_sq_dist) if median_sq_dist > 0 else 1.0
    return gamma

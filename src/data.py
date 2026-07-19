import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


def make_synthetic_classification_dataset(n_samples=400, c=1.5, std=0.6, random_state=0, test_size=0.3):
    rng = np.random.default_rng(random_state)
    n_per_cluster = n_samples // 4
    
    cluster1 = rng.normal(loc=[-c, -c], scale=std, size=(n_per_cluster, 2))
    cluster2 = rng.normal(loc=[c, -c], scale=std, size=(n_per_cluster, 2))
    cluster3 = rng.normal(loc=[-c, c], scale=std, size=(n_per_cluster, 2))
    cluster4 = rng.normal(loc=[c, c], scale=std, size=(n_per_cluster, 2))

    x = np.vstack((cluster1, cluster2, cluster3, cluster4))
    y = np.array([-1] * n_per_cluster + [1] * n_per_cluster + [1] * n_per_cluster + [-1] * n_per_cluster)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform sul train
    X_test = scaler.transform(X_test)         # solo transform sul test, usando le statistiche del train

    return X_train, X_test, y_train, y_test



def load_real_world_dataset(test_size=0.3, random_state=0):
    data = load_digits()

    X = data.data.astype(np.float64)
    y = np.where(data.target % 2 == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

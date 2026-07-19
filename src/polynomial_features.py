from sklearn.preprocessing import PolynomialFeatures


class PolynomialFeatureMap:
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X):
        self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.poly_.fit(X)
        return self

    def transform(self, X):
        return self.poly_.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
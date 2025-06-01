import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[idx]
        self.components = np.real(eigenvectors[:self.n_components])

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        return (X - self.mean) @ self.components.T

    def reconstruct(self, X):
        #TODO: 2%
        return (X - self.mean) @ self.components.T @ self.components + self.mean
        

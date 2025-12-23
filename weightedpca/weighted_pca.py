"""Weighted Principal Component Analysis."""

import numpy as np
from scipy import linalg


class WeightedPCA:
    """Weighted Principal Component Analysis."""

    def __init__(self, n_components=None, scale=False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, sample_weight=None):
        """Fit the model."""
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Handle weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        sample_weight = np.asarray(sample_weight)

        # Weighted mean
        self.mean_ = np.average(X, axis=0, weights=sample_weight)
        X_centered = X - self.mean_

        # Optional scaling
        if self.scale:
            self.scale_ = self._weighted_std(X_centered, sample_weight)
            X_centered = X_centered / self.scale_
        else:
            self.scale_ = None

        # Weighted covariance
        cov = self._weighted_cov(X_centered, sample_weight, n_samples)

        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        n_comp = self.n_components or min(n_samples, n_features)
        self.n_components_ = n_comp
        self.components_ = eigenvectors[:, :n_comp].T
        self.explained_variance_ = eigenvalues[:n_comp]

        # Compute ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def _weighted_cov(self, X_centered, sample_weight, n_samples):
        """Compute weighted covariance matrix."""
        sqrt_w = np.sqrt(sample_weight)
        X_weighted = X_centered * sqrt_w[:, np.newaxis]
        sum_w = np.sum(sample_weight)
        cov = X_weighted.T @ X_weighted / sum_w * (n_samples / (n_samples - 1))
        return cov

    def _weighted_std(self, X_centered, sample_weight):
        """Compute weighted standard deviation."""
        sum_w = np.sum(sample_weight)
        variance = np.sum(sample_weight[:, np.newaxis] * X_centered**2, axis=0) / sum_w
        std = np.sqrt(variance)
        # Replace zero std with 1 to avoid division by zero (constant features stay unchanged)
        std[std == 0] = 1.0
        return std

    def transform(self, X):
        """Project X onto principal components."""
        X = np.asarray(X)
        X_centered = X - self.mean_
        if self.scale_ is not None:
            X_centered = X_centered / self.scale_
        return X_centered @ self.components_.T

    def fit_transform(self, X, sample_weight=None):
        """Fit and transform."""
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Transform back to original space."""
        X_centered = X_transformed @ self.components_
        if self.scale_ is not None:
            X_centered = X_centered * self.scale_
        return X_centered + self.mean_

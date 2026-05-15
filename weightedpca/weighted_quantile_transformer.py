"""Weighted Quantile Transformer."""

import numpy as np
from scipy import stats


class WeightedQuantileTransformer:
    """Transform features using weighted quantiles information.

    This transformer maps features to a uniform or normal distribution using
    weighted quantiles. It is similar to sklearn.preprocessing.QuantileTransformer
    but accepts sample weights.

    If one sample has weight p while all others have weight 1, the results are
    identical to including that sample p times in the data. Scaling all weights
    by a constant has no effect on the transformation. If all weights are equal,
    the results are identical to sklearn's QuantileTransformer.

    Parameters
    ----------
    n_quantiles : int, default=1000
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    subsample : int or None, default=10000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ from the original sklearn implementation.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling. Pass an int for
        reproducible results across multiple function calls.

    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    Attributes
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles_, n_features)
        The values corresponding to the quantiles of reference.

    references_ : ndarray of shape (n_quantiles_,)
        Quantiles of references (uniform or normal distribution).

    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        subsample=10000,
        random_state=None,
        copy=True,
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def fit(self, X, weights=None):
        """Fit the weighted quantile transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the weighted quantiles.

        weights : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples have
            equal weight of 1.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Store training data and weights for later use (needed for scale-invariant transform)
        self.X_train_ = X.copy()

        # Handle weights
        if weights is None:
            weights = np.ones(n_samples)
        else:
            weights = np.asarray(weights).copy()

        # Store weights for transform
        self.weights_train_ = weights.copy()

        # Validate weights
        if weights.shape[0] != n_samples:
            raise ValueError(
                f"weights has {weights.shape[0]} samples "
                f"but X has {n_samples} samples"
            )
        if np.any(weights < 0):
            raise ValueError("weights cannot contain negative values")
        if np.sum(weights) == 0:
            raise ValueError("sum of weights must be positive")

        # Normalize weights by dividing by the minimum non-zero weight
        min_weight = np.min(weights[weights > 0])
        weights = weights / min_weight

        # Handle subsampling
        if self.subsample is not None and n_samples > self.subsample:
            X_sub, weights_sub = self._subsample(X, weights, self.subsample)
            # For transform, use the subsampled data
            self.X_train_ = X_sub.copy()
            self.weights_train_ = weights_sub.copy()
            X = X_sub
            weights = weights_sub
            n_samples = X.shape[0]

        # Use total weight as the virtual sample count
        total_weight = np.sum(weights)

        # Determine actual number of quantiles
        # Always use n_quantiles as specified, capped at the number of distinct samples
        # This ensures weight scaling doesn't affect the quantile grid.
        # The mapping from quantile levels (0, 1) to virtual positions uses total_weight,
        # so scaling weights automatically scales virtual positions while keeping the
        # quantile grid the same, preserving scaling invariance.
        self.n_quantiles_ = min(self.n_quantiles, n_samples)

        # Compute reference quantiles based on output distribution
        self.references_ = self._get_reference_quantiles(self.n_quantiles_)

        # Compute weighted quantiles for each feature
        self.quantiles_ = np.zeros((self.n_quantiles_, n_features))

        for feature_idx in range(n_features):
            self.quantiles_[:, feature_idx] = self._compute_weighted_quantiles(
                X[:, feature_idx], weights, total_weight, self.references_
            )

        return self

    def transform(self, X):
        """Transform X using the weighted quantile mapping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.copy:
            X = X.copy()

        n_samples, n_features = X.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features but transformer was fitted "
                f"with {self.n_features_in_} features"
            )

        X_transformed = np.zeros_like(X, dtype=float)

        # For each feature, transform via the weighted quantiles
        for feature_idx in range(n_features):
            X_transformed[:, feature_idx] = self._transform_feature(
                X[:, feature_idx], self.quantiles_[:, feature_idx], self.references_
            )

        if self.output_distribution == "normal":
            bounds_threshold = 1e-7
            X_transformed = np.clip(
                X_transformed, bounds_threshold, 1 - bounds_threshold
            )
            X_transformed = stats.norm.ppf(X_transformed)

        return X_transformed

    def fit_transform(self, X, weights=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the weighted quantiles and to transform.

        weights : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        return self.fit(X, weights=weights).transform(X)

    def inverse_transform(self, X):
        """Transform back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            The inverse transformed data.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.copy:
            X = X.copy()

        n_samples, n_features = X.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features but transformer was fitted "
                f"with {self.n_features_in_} features"
            )

        X_original = np.zeros_like(X, dtype=float)

        if self.output_distribution == "normal":
            X = stats.norm.cdf(X)

        # Inverse transform each feature
        for feature_idx in range(n_features):
            X_original[:, feature_idx] = self._inverse_transform_feature(
                X[:, feature_idx], self.quantiles_[:, feature_idx], self.references_
            )

        return X_original

    def _compute_weighted_quantiles(self, x, weights, total_weight, quantiles):
        """Compute weighted quantiles for a single feature.

        This implements weighted quantiles that match numpy.quantile behavior
        when weights correspond to repeated samples.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Feature values.

        weights : array-like of shape (n_samples,)
            Sample weights (normalized).

        total_weight : float
            Sum of all weights.

        quantiles : array-like of shape (n_quantiles,)
            Quantiles to compute (between 0 and 1).

        Returns
        -------
        weighted_quantiles : ndarray of shape (n_quantiles,)
            The weighted quantile values.
        """
        # Sort by feature value
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        weights_sorted = weights[sorted_indices]

        # Build virtual index boundaries
        # Sample i with weight w_i maps to virtual indices [cumsum[i-1], cumsum[i])
        cumsum = np.concatenate([[0], np.cumsum(weights_sorted)])

        # For each quantile, compute the virtual position using numpy's formula
        weighted_quantiles = np.zeros(len(quantiles))

        for i, q in enumerate(quantiles):
            # Virtual position using numpy.quantile formula: q * (n - 1)
            # where n is the total virtual sample count (sum of weights)
            virtual_position = q * (total_weight - 1)

            # We need to interpolate between the value at floor(virtual_position)
            # and ceil(virtual_position)
            v_left = np.floor(virtual_position)
            v_right = np.ceil(virtual_position)
            f = virtual_position - v_left

            # Find idx for v_left
            idx_left = np.searchsorted(cumsum, v_left, side="right") - 1
            idx_left = max(0, min(idx_left, len(x_sorted) - 1))
            val_left = x_sorted[idx_left]

            # Find idx for v_right
            idx_right = np.searchsorted(cumsum, v_right, side="right") - 1
            idx_right = max(0, min(idx_right, len(x_sorted) - 1))
            val_right = x_sorted[idx_right]

            # Interpolate
            weighted_quantiles[i] = val_left + f * (val_right - val_left)

        return weighted_quantiles

    def _transform_feature(self, x, quantiles, references):
        """Transform a single feature using quantile mapping.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Feature values to transform.

        quantiles : array-like of shape (n_quantiles,)
            Quantile values from training data.

        references : array-like of shape (n_quantiles,)
            Reference quantiles to map to.

        Returns
        -------
        x_transformed : ndarray of shape (n_samples,)
            Transformed feature values.
        """
        # Use piecewise linear interpolation
        # Handle values outside the training range
        x_transformed = np.interp(x, quantiles, references)
        return x_transformed

    def _inverse_transform_feature(self, x, quantiles, references):
        """Inverse transform a single feature.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            Transformed feature values.

        quantiles : array-like of shape (n_quantiles,)
            Quantile values from training data.

        references : array-like of shape (n_quantiles,)
            Reference quantiles used in forward transform.

        Returns
        -------
        x_original : ndarray of shape (n_samples,)
            Original feature values.
        """
        # Inverse is just swapping the roles of quantiles and references
        x_original = np.interp(x, references, quantiles)
        return x_original

    def _get_reference_quantiles(self, n_quantiles):
        """Get reference quantiles for the target distribution.

        Parameters
        ----------
        n_quantiles : int
            Number of quantiles.

        Returns
        -------
        references : ndarray of shape (n_quantiles,)
            Reference quantile values.
        """
        # Quantile positions (use midpoints to avoid 0 and 1 exactly for normal)
        quantile_positions = np.linspace(0, 1, n_quantiles)

        if self.output_distribution not in ["uniform", "normal"]:
            raise ValueError(
                f"output_distribution must be 'uniform' or 'normal', "
                f"got '{self.output_distribution}'"
            )

        return quantile_positions

    def _subsample(self, X, weights, n_subsample):
        """Subsample the data while respecting weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        weights : array-like of shape (n_samples,)
            Sample weights.

        n_subsample : int
            Number of samples to keep.

        Returns
        -------
        X_subsample : ndarray of shape (n_subsample, n_features)
            Subsampled data.

        weights_subsample : ndarray of shape (n_subsample,)
            Subsampled weights (as counts).
        """
        rng = np.random.RandomState(self.random_state)

        # Normalize weights for sampling probabilities
        normalized_weights = weights / np.sum(weights)

        # Sample according to weights
        indices = rng.choice(
            len(X), size=n_subsample, replace=True, p=normalized_weights
        )

        # Count how many times each unique index was selected
        unique_indices, counts = np.unique(indices, return_counts=True)

        X_subsample = X[unique_indices]
        # Return counts (will be renormalized in fit)
        weights_subsample = counts.astype(float)

        return X_subsample, weights_subsample

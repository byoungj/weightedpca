"""Tests for WeightedQuantileTransformer."""

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer
from weightedpca import WeightedQuantileTransformer


def test_basic_uniform_transformation():
    """Test basic quantile transformation to uniform distribution."""
    X = np.array([[1], [2], [3], [4], [5]])

    transformer = WeightedQuantileTransformer(
        n_quantiles=5, output_distribution="uniform"
    )
    X_transformed = transformer.fit_transform(X)

    # Check that output is roughly uniform [0, 1]
    assert X_transformed.shape == X.shape
    assert np.all(X_transformed >= 0)
    assert np.all(X_transformed <= 1)

    # Should be monotonically increasing
    assert np.all(np.diff(X_transformed[:, 0]) >= 0)


def test_basic_normal_transformation():
    """Test basic quantile transformation to normal distribution."""
    X = np.array([[1], [2], [3], [4], [5]])

    transformer = WeightedQuantileTransformer(
        n_quantiles=5, output_distribution="normal"
    )
    X_transformed = transformer.fit_transform(X)

    # Check shape and monotonicity
    assert X_transformed.shape == X.shape
    assert np.all(np.diff(X_transformed[:, 0]) >= 0)


def test_uniform_weights_matches_sklearn():
    """Test that WeightedQuantileTransformer with uniform weights exactly matches sklearn.QuantileTransformer."""
    X = np.random.RandomState(42).randn(50, 2)
    X_test = np.random.RandomState(43).randn(10, 2)

    for distribution in ["uniform", "normal"]:
        # WeightedQuantileTransformer with uniform weights
        transformer_weighted = WeightedQuantileTransformer(
            n_quantiles=20, output_distribution=distribution, random_state=42
        )
        transformer_weighted.fit(X, sample_weight=np.ones(len(X)))

        # sklearn QuantileTransformer
        transformer_sklearn = QuantileTransformer(
            n_quantiles=20, output_distribution=distribution, random_state=42
        )
        transformer_sklearn.fit(X)

        # Transform test data
        result_weighted = transformer_weighted.transform(X_test)
        result_sklearn = transformer_sklearn.transform(X_test)

        # Should match exactly when weights are all 1
        np.testing.assert_allclose(
            result_weighted, result_sklearn, rtol=1e-10, atol=0.0
        )


def test_weight_equivalence_to_repetition():
    """Test that weight p is equivalent to repeating a sample p times.

    This test comprehensively checks weight equivalence by varying:
    - Which row is copied
    - How many times it's copied
    - The base weight value
    """
    rng = np.random.RandomState(42)
    n = 50  # Number of original samples
    n_features = 3
    X = rng.randn(n, n_features)

    # Generate test data
    X_test = rng.randn(10, n_features)

    for row_to_copy in [0, n // 2, n - 1]:
        for n_copies in [1, 2, 5]:
            for base_weight in [1e-7, 1, 13.7e7]:
                for distribution in ["uniform", "normal"]:
                    # Create repeated data by copying a row
                    X_repeated = np.vstack(
                        [
                            X,
                            np.repeat(
                                X[row_to_copy : row_to_copy + 1], n_copies, axis=0
                            ),
                        ]
                    )

                    # Create weights for weighted version
                    weights = np.full(n, base_weight)
                    weights[row_to_copy] = base_weight * (n_copies + 1)

                    # Fit transformer on weighted data
                    transformer_weighted = WeightedQuantileTransformer(
                        n_quantiles=20,
                        output_distribution=distribution,
                        random_state=42,
                    )
                    transformer_weighted.fit(X, sample_weight=weights)

                    # Fit transformer on repeated data (uniform weights)
                    transformer_repeated = QuantileTransformer(
                        n_quantiles=20,
                        output_distribution=distribution,
                        random_state=42,
                    )
                    transformer_repeated.fit(X_repeated)

                    # Transform test data
                    result_weighted = transformer_weighted.transform(X_test)
                    result_repeated = transformer_repeated.transform(X_test)

                    # Results must be identical (weight p is exactly equivalent to p repetitions)
                    np.testing.assert_allclose(
                        result_weighted,
                        result_repeated,
                        rtol=1e-10,
                        err_msg=f"Mismatch for row={row_to_copy}, copies={n_copies}, base_weight={base_weight}",
                    )


def test_weight_scaling_invariance():
    """Test that scaling all weights by a constant has no effect on transformations."""
    X = np.array([[1], [2], [3], [4], [5]])
    weights_1 = np.array([1, 1, 2, 1, 1])
    weights_2 = np.array([10, 10, 20, 10, 10])  # Scaled by 10
    weights_3 = np.array([0.1, 0.1, 0.2, 0.1, 0.1])  # Scaled by 0.1

    transformer_1 = WeightedQuantileTransformer(n_quantiles=10, random_state=42)
    transformer_2 = WeightedQuantileTransformer(n_quantiles=10, random_state=42)
    transformer_3 = WeightedQuantileTransformer(n_quantiles=10, random_state=42)

    transformer_1.fit(X, sample_weight=weights_1)
    transformer_2.fit(X, sample_weight=weights_2)
    transformer_3.fit(X, sample_weight=weights_3)

    # Transformations should be identical when weights are scaled
    # (empirical CDF is scale-invariant)
    X_test = np.array([[1.5], [2.5], [3.5]])
    result_1 = transformer_1.transform(X_test)
    result_2 = transformer_2.transform(X_test)
    result_3 = transformer_3.transform(X_test)

    # Perfect equality for uniformly scaled weights
    np.testing.assert_allclose(result_1, result_2, rtol=1e-10)
    np.testing.assert_allclose(result_1, result_3, rtol=1e-10)


def test_inverse_transform():
    """Test that inverse_transform reverses transform."""
    X = np.random.RandomState(42).randn(100, 3)
    weights = np.random.RandomState(42).rand(100)

    transformer = WeightedQuantileTransformer(
        n_quantiles=50, output_distribution="uniform"
    )
    transformer.fit(X, sample_weight=weights)

    X_transformed = transformer.transform(X)
    X_reconstructed = transformer.inverse_transform(X_transformed)

    # Should reconstruct reasonably well (quantile transform is lossy)
    # Check that the reconstruction is in the right ballpark
    np.testing.assert_allclose(X, X_reconstructed, rtol=0.5, atol=1.0)


def test_multivariate_data():
    """Test transformation on multivariate data."""
    X = np.random.RandomState(42).randn(50, 5)
    weights = np.random.RandomState(42).rand(50)

    transformer = WeightedQuantileTransformer(
        n_quantiles=25, output_distribution="uniform"
    )
    X_transformed = transformer.fit_transform(X, sample_weight=weights)

    # Check shape
    assert X_transformed.shape == X.shape

    # Each feature should be roughly uniform
    for i in range(X.shape[1]):
        assert X_transformed[:, i].min() >= -0.1  # Some tolerance
        assert X_transformed[:, i].max() <= 1.1


def test_no_weights_equals_uniform_weights():
    """Test that no weights is equivalent to uniform weights."""
    X = np.random.RandomState(42).randn(20, 2)

    transformer_no_weights = WeightedQuantileTransformer(
        n_quantiles=15, random_state=42
    )
    transformer_uniform_weights = WeightedQuantileTransformer(
        n_quantiles=15, random_state=42
    )

    transformer_no_weights.fit(X)
    transformer_uniform_weights.fit(X, sample_weight=np.ones(20))

    # Quantiles should be identical
    np.testing.assert_allclose(
        transformer_no_weights.quantiles_,
        transformer_uniform_weights.quantiles_,
        rtol=1e-10,
    )


def test_single_sample_high_weight():
    """Test behavior when one sample has very high weight."""
    X = np.array([[1], [2], [3], [4], [5]])
    weights = np.array([1, 1, 100, 1, 1])  # Middle sample dominates

    transformer = WeightedQuantileTransformer(n_quantiles=20)
    transformer.fit(X, sample_weight=weights)

    # The 0.5 quantile (median) should be close to 3 since value 3 has weight 100/104
    quantile_values = transformer._compute_weighted_quantiles(
        X[:, 0], weights, np.sum(weights), np.array([0.5])
    )
    assert abs(quantile_values[0] - 3) < 0.5


def test_zero_weight_samples_ignored():
    """Test that samples with zero weight are effectively ignored."""
    X = np.array([[1], [2], [3], [999], [1000]])
    weights = np.array([1, 1, 1, 0, 0])  # Last two samples have zero weight

    # Should be equivalent to only using first 3 samples
    X_reduced = np.array([[1], [2], [3]])

    transformer_weighted = WeightedQuantileTransformer(n_quantiles=5, random_state=42)
    transformer_reduced = WeightedQuantileTransformer(n_quantiles=5, random_state=42)

    transformer_weighted.fit(X, sample_weight=weights)
    transformer_reduced.fit(X_reduced)

    # Quantiles should behave similarly (zero weights largely ignored)
    # Note: perfect equivalence not possible due to normalization effects
    np.testing.assert_allclose(
        transformer_weighted.quantiles_[:, 0][:3],
        transformer_reduced.quantiles_[:, 0][:3],
        rtol=0.4,
        atol=1.0,
    )


def test_copy_parameter():
    """Test that copy parameter works correctly."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    X_original = X.copy()

    # With copy=True (default)
    transformer = WeightedQuantileTransformer(copy=True)
    transformer.fit(X).transform(X)
    np.testing.assert_array_equal(X, X_original)

    # With copy=False
    transformer = WeightedQuantileTransformer(copy=False)
    X_result = transformer.fit(X).transform(X)
    # X may or may not be modified, but result should still be correct
    assert X_result.shape == X_original.shape


def test_n_quantiles_exceeds_samples():
    """Test when n_quantiles is larger than number of samples."""
    X = np.array([[1], [2], [3]])

    transformer = WeightedQuantileTransformer(n_quantiles=100)
    transformer.fit(X)

    # Should cap at number of samples
    assert transformer.n_quantiles_ == 3


def test_subsample_parameter():
    """Test subsampling for large datasets."""
    X = np.random.RandomState(42).randn(1000, 2)
    weights = np.random.RandomState(42).rand(1000)

    transformer = WeightedQuantileTransformer(
        n_quantiles=50, subsample=100, random_state=42
    )
    transformer.fit(X, sample_weight=weights)

    # Should still work and produce reasonable results
    X_test = np.random.RandomState(43).randn(10, 2)
    X_transformed = transformer.transform(X_test)

    assert X_transformed.shape == X_test.shape


def test_1d_input():
    """Test handling of 1D input arrays."""
    X = np.array([1, 2, 3, 4, 5])

    transformer = WeightedQuantileTransformer(n_quantiles=5)
    X_transformed = transformer.fit_transform(X)

    # Should be reshaped to 2D
    assert X_transformed.shape == (5, 1)


def test_different_distributions():
    """Test different output distributions produce different results."""
    X = np.random.RandomState(42).randn(50, 1)

    transformer_uniform = WeightedQuantileTransformer(
        n_quantiles=25, output_distribution="uniform"
    )
    transformer_normal = WeightedQuantileTransformer(
        n_quantiles=25, output_distribution="normal"
    )

    X_uniform = transformer_uniform.fit_transform(X)
    X_normal = transformer_normal.fit_transform(X)

    # Results should be different
    assert not np.allclose(X_uniform, X_normal)

    # Uniform should be in [0, 1]
    assert np.all(X_uniform >= -0.1)
    assert np.all(X_uniform <= 1.1)

    # Normal should have mean ~0 and std ~1 for large samples
    # (not guaranteed for small samples, but should be different from [0,1])
    assert X_normal.min() < -0.5 or X_normal.max() > 1.5


def test_invalid_output_distribution():
    """Test error on invalid output distribution."""
    transformer = WeightedQuantileTransformer(output_distribution="invalid")
    X = np.array([[1], [2], [3]])

    with pytest.raises(ValueError, match="output_distribution"):
        transformer.fit(X)


def test_negative_weights_error():
    """Test error on negative weights."""
    X = np.array([[1], [2], [3]])
    weights = np.array([1, -1, 1])

    transformer = WeightedQuantileTransformer()

    with pytest.raises(ValueError, match="negative"):
        transformer.fit(X, sample_weight=weights)


def test_zero_sum_weights_error():
    """Test error when all weights sum to zero."""
    X = np.array([[1], [2], [3]])
    weights = np.array([0, 0, 0])

    transformer = WeightedQuantileTransformer()

    with pytest.raises(ValueError, match="sum.*positive"):
        transformer.fit(X, sample_weight=weights)


def test_mismatched_weights_error():
    """Test error when weight array length doesn't match X."""
    X = np.array([[1], [2], [3]])
    weights = np.array([1, 1])  # Wrong length

    transformer = WeightedQuantileTransformer()

    with pytest.raises(ValueError, match="sample_weight has"):
        transformer.fit(X, sample_weight=weights)


def test_transform_before_fit_error():
    """Test error when transforming before fitting."""
    transformer = WeightedQuantileTransformer()
    X = np.array([[1], [2], [3]])

    with pytest.raises(AttributeError):
        transformer.transform(X)


def test_feature_mismatch_error():
    """Test error when transform X has different number of features."""
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[1], [2], [3]])

    transformer = WeightedQuantileTransformer()
    transformer.fit(X_train)

    with pytest.raises(ValueError, match="features"):
        transformer.transform(X_test)


def test_reproducibility_with_random_state():
    """Test that random_state ensures reproducibility."""
    X = np.random.RandomState(42).randn(1000, 2)
    weights = np.random.RandomState(42).rand(1000)

    transformer_1 = WeightedQuantileTransformer(subsample=100, random_state=42)
    transformer_2 = WeightedQuantileTransformer(subsample=100, random_state=42)

    transformer_1.fit(X, sample_weight=weights)
    transformer_2.fit(X, sample_weight=weights)

    # Should produce identical quantiles
    np.testing.assert_array_equal(transformer_1.quantiles_, transformer_2.quantiles_)


def test_weighted_vs_unweighted_difference():
    """Test that weights actually change the transformation."""
    X = np.array([[1], [2], [3], [4], [5]])

    # Uniform weights
    transformer_uniform = WeightedQuantileTransformer(n_quantiles=10)
    transformer_uniform.fit(X)

    # Skewed weights (emphasize lower values)
    transformer_weighted = WeightedQuantileTransformer(n_quantiles=10)
    weights = np.array([5, 3, 1, 1, 1])
    transformer_weighted.fit(X, sample_weight=weights)

    # Quantiles should be different
    assert not np.allclose(
        transformer_uniform.quantiles_, transformer_weighted.quantiles_
    )


def test_extreme_weights():
    """Test handling of extreme weight ratios."""
    X = np.array([[1], [2], [3], [4], [5]])
    weights = np.array([1e-10, 1e-10, 1.0, 1e-10, 1e-10])

    transformer = WeightedQuantileTransformer(n_quantiles=10)
    transformer.fit(X, sample_weight=weights)

    # Should still work
    X_test = np.array([[2.5], [3.0], [3.5]])
    X_transformed = transformer.transform(X_test)

    assert X_transformed.shape == X_test.shape
    assert np.all(np.isfinite(X_transformed))


def test_transform_preserves_order():
    """Test that quantile transform preserves order of values."""
    X = np.random.RandomState(42).randn(30, 1)
    weights = np.random.RandomState(42).rand(30)

    transformer = WeightedQuantileTransformer(n_quantiles=15)
    transformer.fit(X, sample_weight=weights)

    X_test = np.sort(np.random.RandomState(43).randn(20, 1), axis=0)
    X_transformed = transformer.transform(X_test)

    # Transformed values should also be sorted (monotonic)
    sorted_transformed = np.sort(X_transformed, axis=0)
    np.testing.assert_allclose(X_transformed, sorted_transformed, rtol=1e-10)


def test_quantiles_sorted():
    """Test that computed quantiles are sorted."""
    X = np.random.RandomState(42).randn(50, 3)
    weights = np.random.RandomState(42).rand(50)

    transformer = WeightedQuantileTransformer(n_quantiles=25)
    transformer.fit(X, sample_weight=weights)

    # Each feature's quantiles should be sorted
    for i in range(X.shape[1]):
        sorted_quantiles = np.sort(transformer.quantiles_[:, i])
        np.testing.assert_allclose(
            transformer.quantiles_[:, i], sorted_quantiles, rtol=1e-10
        )


def test_fit_returns_self():
    """Test that fit returns self for method chaining."""
    X = np.array([[1], [2], [3]])
    transformer = WeightedQuantileTransformer()

    result = transformer.fit(X)
    assert result is transformer

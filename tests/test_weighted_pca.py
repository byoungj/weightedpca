"""Tests for WeightedPCA."""

import numpy as np
from sklearn.decomposition import PCA
from weightedpca import WeightedPCA

def test_it_runs():
    """Just check it doesn't crash."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    wpca = WeightedPCA(n_components=2)
    wpca.fit(X)

    assert wpca.components_ is not None


def test_shapes():
    """Test output shapes."""
    X = np.random.randn(100, 10)
    wpca = WeightedPCA(n_components=5)
    X_t = wpca.fit_transform(X)

    assert X_t.shape == (100, 5)
    assert wpca.components_.shape == (5, 10)
    assert wpca.mean_.shape == (10,)


def test_inverse_transform():
    """Roundtrip should recover original."""
    X = np.random.randn(50, 5)
    wpca = WeightedPCA(n_components=5)
    X_t = wpca.fit_transform(X)
    X_back = wpca.inverse_transform(X_t)

    np.testing.assert_allclose(X, X_back, rtol=1e-10)


def test_uniform_weights_equals_no_weights():
    """Uniform weights should give same result as no weights."""
    rng = np.random.RandomState(42)
    n = 50
    X = rng.randn(n, 10)
    for weight in [2e-7, 1, 13.7e6]:
        weights = np.full(n, weight)

        wpca_no_weights = WeightedPCA(n_components=5)
        wpca_no_weights.fit(X)

        wpca_uniform = WeightedPCA(n_components=5)
        wpca_uniform.fit(X, sample_weight=weights)

        np.testing.assert_allclose(wpca_no_weights.mean_, wpca_uniform.mean_)
        np.testing.assert_allclose(
            np.abs(wpca_no_weights.components_),
            np.abs(wpca_uniform.components_),
            rtol=1e-10,
        )


def test_weights_affect_mean():
    """Weighted mean should differ from unweighted mean."""
    X = np.array([[0, 0], [10, 10]])
    weights = np.array([1.0, 9.0])  # heavily weight second sample

    wpca = WeightedPCA(n_components=2)
    wpca.fit(X, sample_weight=weights)

    # Weighted mean should be closer to [10, 10]
    np.testing.assert_allclose(wpca.mean_, [9.0, 9.0])


def test_weights_change_components():
    """Different weights should give different components."""
    rng = np.random.RandomState(123)
    X = rng.randn(100, 5)
    w1 = np.ones(100)
    w2 = rng.uniform(0.1, 10, 100)

    wpca1 = WeightedPCA(n_components=3)
    wpca1.fit(X, sample_weight=w1)

    wpca2 = WeightedPCA(n_components=3)
    wpca2.fit(X, sample_weight=w2)

    # Components should NOT be equal
    assert not np.allclose(wpca1.components_, wpca2.components_)


def test_explained_variance():
    """Should have explained_variance_ratio_ attribute."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)

    wpca = WeightedPCA(n_components=5)
    wpca.fit(X)

    assert hasattr(wpca, "explained_variance_ratio_")
    assert wpca.explained_variance_ratio_.shape == (5,)
    assert np.all(wpca.explained_variance_ratio_ >= 0)
    assert np.sum(wpca.explained_variance_ratio_) <= 1.0 + 1e-10


def test_matches_sklearn_pca():
    """With uniform weights, should match sklearn PCA."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)

    pca = PCA(n_components=5)
    pca.fit(X)

    wpca = WeightedPCA(n_components=5)
    wpca.fit(X)

    np.testing.assert_allclose(pca.mean_, wpca.mean_, rtol=1e-10)
    np.testing.assert_allclose(
        np.abs(pca.components_), np.abs(wpca.components_), rtol=1e-5
    )
    np.testing.assert_allclose(
        pca.explained_variance_ratio_,
        wpca.explained_variance_ratio_,
        rtol=1e-5,
    )

def test_weight_matches_copies():
    """Adding copies of a row should be equivalent to increasing the weight."""
    rng = np.random.RandomState(42)
    n = 100  # Number of original samples
    p = 10  # Number of features
    X = rng.randn(n, p)
    # Make some data to transform:
    X_to_transform = rng.randn(20, p)

    for row_to_copy in [0, n//2, n-1]:
        for n_copies in [1, 2, 5]:
            for base_weight in [1e-7, 1, 13.7e7]:
                X_with_copies = np.concatenate(
                    [
                        X,
                        np.repeat(X[row_to_copy:row_to_copy+1], n_copies, axis=0)
                    ]
                )
                weights = np.full(n, base_weight)
                weights[row_to_copy] = base_weight * (n_copies + 1)  # increase weight of row that was copied

                pca = PCA()
                pca.fit(X_with_copies)

                wpca = WeightedPCA()
                wpca.fit(X, sample_weight=weights)

                np.testing.assert_allclose(pca.mean_, wpca.mean_, rtol=1e-10)
                np.testing.assert_allclose(
                    pca.explained_variance_ratio_,
                    wpca.explained_variance_ratio_,
                    rtol=1e-10,
                )
                # Components may differ by sign
                np.testing.assert_allclose(
                    np.abs(pca.components_), np.abs(wpca.components_), rtol=1e-10
                )
                # More granular test: each component should agree up to overall sign
                for j_component in range(p):
                    assert np.allclose(
                        pca.components_[j_component], wpca.components_[j_component], rtol=1e-10
                    ) or np.allclose(
                        pca.components_[j_component], -wpca.components_[j_component], rtol=1e-10
                    ), f"Component {j_component} does not match"

                # Check that transformed data also matches
                Y_pca = pca.transform(X_to_transform)
                Y_wpca = wpca.transform(X_to_transform)
                for j_component in range(p):
                    # Components may differ by sign
                    assert np.allclose(
                        Y_pca[:, j_component], Y_wpca[:, j_component], rtol=1e-10
                    ) or np.allclose(
                        Y_pca[:, j_component], -Y_wpca[:, j_component], rtol=1e-10
                    ), f"Transformed component {j_component} does not match"

                # Check that inverse-transformed data also matches
                X_pca_back = pca.inverse_transform(Y_pca)
                X_wpca_back = wpca.inverse_transform(Y_wpca)
                np.testing.assert_allclose(X_pca_back, X_wpca_back, rtol=1e-10)

def test_scaling():
    """Scaling should normalize feature variances."""
    rng = np.random.RandomState(42)
    # Create data with very different scales
    X = np.column_stack([
        rng.randn(100) * 1,      # small variance
        rng.randn(100) * 1000,   # large variance
    ])

    # Without scaling, large-variance feature dominates
    wpca_no_scale = WeightedPCA(n_components=2, scale=False)
    wpca_no_scale.fit(X)

    # With scaling, features are comparable
    wpca_scaled = WeightedPCA(n_components=2, scale=True)
    wpca_scaled.fit(X)

    # Check scale_ attribute is set
    assert wpca_scaled.scale_ is not None
    assert wpca_scaled.scale_.shape == (2,)

    # With scaling, first PC should not be dominated by feature 1
    # (components should be more balanced)
    assert not np.allclose(wpca_no_scale.components_, wpca_scaled.components_)

    # Roundtrip should still work with scaling
    X_t = wpca_scaled.transform(X)
    X_back = wpca_scaled.inverse_transform(X_t)
    np.testing.assert_allclose(X, X_back, rtol=1e-10)

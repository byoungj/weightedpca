"""Tests for WeightedPCA."""

import numpy as np

def test_it_runs():
    """Just check it doesn't crash."""
    from weightedpca import WeightedPCA

    X = np.array([[1, 2], [3, 4], [5, 6]])
    wpca = WeightedPCA(n_components=2)
    wpca.fit(X)

    assert wpca.components_ is not None


def test_shapes():
    """Test output shapes."""
    from weightedpca import WeightedPCA

    X = np.random.randn(100, 10)
    wpca = WeightedPCA(n_components=5)
    X_t = wpca.fit_transform(X)

    assert X_t.shape == (100, 5)
    assert wpca.components_.shape == (5, 10)
    assert wpca.mean_.shape == (10,)


def test_inverse_transform():
    """Roundtrip should recover original."""
    from weightedpca import WeightedPCA

    X = np.random.randn(50, 5)
    wpca = WeightedPCA(n_components=5)
    X_t = wpca.fit_transform(X)
    X_back = wpca.inverse_transform(X_t)

    np.testing.assert_allclose(X, X_back, rtol=1e-10)


def test_uniform_weights_equals_no_weights():
    """Uniform weights should give same result as no weights."""
    from weightedpca import WeightedPCA

    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    weights = np.ones(50)

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
    from weightedpca import WeightedPCA

    X = np.array([[0, 0], [10, 10]])
    weights = np.array([1.0, 9.0])  # heavily weight second sample

    wpca = WeightedPCA(n_components=2)
    wpca.fit(X, sample_weight=weights)

    # Weighted mean should be closer to [10, 10]
    np.testing.assert_allclose(wpca.mean_, [9.0, 9.0])


def test_weights_change_components():
    """Different weights should give different components."""
    from weightedpca import WeightedPCA

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
    from weightedpca import WeightedPCA

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
    from sklearn.decomposition import PCA
    from weightedpca import WeightedPCA

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


def test_scaling():
    """Scaling should normalize feature variances."""
    from weightedpca import WeightedPCA

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

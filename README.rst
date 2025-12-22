###########
WeightedPCA
###########

**A scikit-learn compatible implementation of Weighted Principal Component Analysis**

|License| |sklearn|

Overview
========

WeightedPCA is an extension to scikit-learn that implements Weighted Principal Component Analysis. 
It follows the scikit-learn API conventions, making it a drop-in replacement for 
``sklearn.decomposition.PCA`` when you need to assign different weights to your samples.

This is a **simplified implementation** that supports **sample-wise (row) weighting only**. 
Each sample can have a different weight, but all features within a sample share the same weight.
For full element-wise weighting (where each individual measurement can have its own weight), 
see the `wpca package <https://github.com/jakevdp/wpca>`__, which implements 
the complete Delchambre (2014) algorithm.

This package is **not part of scikit-learn**, but is designed to be fully compatible with 
the scikit-learn ecosystem, including:

- Following scikit-learn's ``fit``/``transform``/``fit_transform`` API
- Inheriting from scikit-learn's ``_BasePCA``
- Using scikit-learn's validation utilities

Installation
============

.. code-block:: console

    pip install weightedpca

Or install from source:

.. code-block:: console

    git clone https://github.com/byoungj/weightedpca.git
    cd weightedpca
    pip install -e .


Quick Start
===========

.. code-block:: python

    import numpy as np
    from weightedpca import WeightedPCA

    # Your data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Sample weights (e.g., based on data quality or importance)
    weights = np.array([1.0, 1.0, 2.0, 2.0])

    # Fit weighted PCA
    wpca = WeightedPCA(n_components=2)
    wpca.fit(X, sample_weight=weights)

    # Transform data
    X_transformed = wpca.transform(X)

    # Or use fit_transform
    X_transformed = wpca.fit_transform(X, sample_weight=weights)


Usage with scikit-learn Pipelines
=================================

WeightedPCA can be used in pipelines. Note that ``sample_weight`` should be 
applied during the WeightedPCA fitting step before the pipeline:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from weightedpca import WeightedPCA

    # Fit WeightedPCA separately with sample weights
    wpca = WeightedPCA(n_components=10)
    X_reduced = wpca.fit_transform(X, sample_weight=weights)

    # Then use pipeline for downstream processing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(X_reduced, y)


Comparison with Standard PCA
============================

.. code-block:: python

    from sklearn.decomposition import PCA
    from weightedpca import WeightedPCA
    import numpy as np

    X = np.random.randn(100, 10)
    weights = np.random.uniform(0.5, 1.5, 100)

    # Standard PCA (unweighted)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)

    # Weighted PCA
    wpca = WeightedPCA(n_components=5)
    X_wpca = wpca.fit_transform(X, sample_weight=weights)

    print(f"Standard PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Weighted PCA explained variance: {wpca.explained_variance_ratio_}")


When to Use Weighted PCA
========================

Weighted PCA is useful when:

1. **Class Imbalance**: Weight samples to balance class representation
2. **Importance Differs**: Certain observations should have more influence
3. **Data Quality Varies**: Some samples are more reliable than others
4. **Uncertainty Quantification**: Use inverse variance as weights
5. **Temporal Data**: Weight recent observations more heavily


Relationship to scikit-learn
============================

This package is an **independent extension** to scikit-learn, not part of the core library. 
We follow scikit-learn's API design principles, coding conventions, and documentation standards.

However, we are **not affiliated with or endorsed by** the scikit-learn project. 
For the official scikit-learn PCA implementation, see 
`sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`__.


License
=======

This project is licensed under the BSD 3-Clause License - the same license as scikit-learn. 
See the `LICENSE <LICENSE>`__ file for details.


References
==========

- Delchambre, L. (2014). "Weighted Principal Component Analysis: A Weighted Covariance Eigendecomposition Approach" `arXiv:1412.4533 <https://arxiv.org/abs/1412.4533>`__
- wpca package (full element-wise implementation): https://github.com/jakevdp/wpca
- scikit-learn documentation: https://scikit-learn.org


.. |License| image:: https://img.shields.io/github/license/byoungj/weightedpca?color=blue
    :target: https://github.com/byoungj/weightedpca/blob/main/LICENSE
    :alt: License

.. |sklearn| image:: https://img.shields.io/badge/scikit--learn-compatible-orange.svg
    :target: https://scikit-learn.org
    :alt: scikit-learn compatible

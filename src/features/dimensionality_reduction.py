import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import prince
import gower
import umap


# DATA PREPARATION

def preprocess_for_reduction(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Builds a fully numeric feature matrix ready for PCA / t-SNE from a mixed-type
    DataFrame. Numeric columns are standardized (zero mean, unit variance) and
    categorical columns are one-hot encoded (the resulting dummy columns are
    already binary, so they are left unscaled).
    Params:
        - df: pd.DataFrame, input data; should not contain missing values (drop
          or impute them beforehand)
        - categorical_columns: list[str] | None, columns to one-hot encode
          (None means none)
        - numerical_columns: list[str] | None, columns to standardize
          (None means none); columns not listed in either argument are ignored
    Returns:
        - features: pd.DataFrame, numeric feature matrix (standardized numeric
          columns concatenated with the one-hot dummy columns)
    """
    categorical_columns = categorical_columns or []
    numerical_columns = numerical_columns or []

    numeric_part = df[numerical_columns].copy()
    if numerical_columns:
        scaler = StandardScaler()
        numeric_part[numerical_columns] = scaler.fit_transform(numeric_part[numerical_columns])

    if categorical_columns:
        # One hot codification
        categorical_part = pd.get_dummies(
            df[categorical_columns].astype(str),
            prefix=categorical_columns,
        )
    else:
        categorical_part = pd.DataFrame(index=df.index)

    return pd.concat([numeric_part, categorical_part], axis=1)


# DIMENSIONALITY REDUCTION

def apply_pca(
    X: np.ndarray | pd.DataFrame,
    n_components: int | float = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Fits PCA on a numeric feature matrix and returns the projected data together
    with the fitted estimator.
    Params:
        - X: np.ndarray | pd.DataFrame, numeric feature matrix
        - n_components: int | float, number of components, or the fraction of
          variance to keep when a float in (0, 1)
        - random_state: int, seed for reproducibility
    Returns:
        - X_pca: np.ndarray, projected data of shape (n_samples, n_components)
        - pca: sklearn.decomposition.PCA, fitted estimator (exposes
          `explained_variance_ratio_`)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def apply_tsne(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    perplexity: float = 30,
) -> np.ndarray:
    """
    Fits t-SNE on a numeric feature matrix and returns the projected data. The
    perplexity is automatically capped to a valid value for small datasets (it
    must stay below the number of samples).
    Params:
        - X: np.ndarray | pd.DataFrame, numeric feature matrix
        - n_components: int, dimension of the embedded space (2 or 3 for
          visualization)
        - random_state: int, seed for reproducibility
        - perplexity: float, t-SNE perplexity; capped to (n_samples - 1) / 3 when
          necessary
    Returns:
        - embedding: np.ndarray, projected data of shape (n_samples, n_components)
    """
    n_samples = X.shape[0]
    perplexity = min(perplexity, max(5, (n_samples - 1) / 3))
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    return tsne.fit_transform(X)


def apply_famd(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, prince.FAMD]:
    """
    Fits Factor Analysis of Mixed Data (FAMD) and returns the row coordinates.
    FAMD is the natural extension of PCA/MCA to data with both numeric and
    categorical variables: numeric columns are standardized and categorical
    columns are handled as in MCA, so there is no need to one-hot encode and
    every variable contributes on a comparable scale. This is usually more
    appropriate than PCA-on-one-hot for clinical tables.
    Params:
        - df: pd.DataFrame, input data (must not contain missing values)
        - categorical_columns: list[str] | None, columns treated as categorical
        - numerical_columns: list[str] | None, columns treated as numeric;
          columns not listed in either argument are ignored
        - n_components: int, number of factors to keep
        - random_state: int, seed for reproducibility
    Returns:
        - X_famd: np.ndarray, row coordinates of shape (n_samples, n_components)
        - famd: prince.FAMD, fitted estimator (exposes `eigenvalues_summary` and
          `percentage_of_variance_`)
    """
    categorical_columns = categorical_columns or []
    numerical_columns = numerical_columns or []

    data = df[categorical_columns + numerical_columns].copy()
    # prince infers the variable type from the dtype, so make it explicit.
    data[categorical_columns] = data[categorical_columns].astype(str)
    data[numerical_columns] = data[numerical_columns].astype(float)

    famd = prince.FAMD(n_components=n_components, random_state=random_state)
    famd = famd.fit(data)
    X_famd = famd.row_coordinates(data).to_numpy()
    return X_famd, famd


def gower_distance(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> np.ndarray:
    """
    Computes the Gower distance matrix for mixed-type data. Gower's coefficient
    combines a normalized absolute distance for numeric variables with a simple
    matching (0/1) distance for categorical ones, yielding a sensible pairwise
    distance in [0, 1] for tables that mix both. Useful as a precomputed metric
    for UMAP / t-SNE on clinical data.
    Params:
        - df: pd.DataFrame, input data (must not contain missing values)
        - categorical_columns: list[str] | None, columns treated as categorical
        - numerical_columns: list[str] | None, columns treated as numeric;
          columns not listed in either argument are ignored. The split is passed
          explicitly to Gower rather than inferred from dtypes
    Returns:
        - distance: np.ndarray, symmetric (n_samples, n_samples) distance matrix
          (float)
    """
    categorical_columns = categorical_columns or []
    numerical_columns = numerical_columns or []

    columns = categorical_columns + numerical_columns
    data = df[columns].copy()
    data[categorical_columns] = data[categorical_columns].astype(str)
    data[numerical_columns] = data[numerical_columns].astype(float)

    cat_features = np.array([col in categorical_columns for col in columns])
    distance = gower.gower_matrix(data, cat_features=cat_features)
    return distance.astype(float)


def apply_umap(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Fits UMAP and returns the embedded coordinates.
    Params:
        - X: np.ndarray | pd.DataFrame, feature matrix, or a precomputed
          (n_samples, n_samples) distance matrix when metric='precomputed'
          (e.g. the output of gower_distance)
        - n_components: int, dimension of the embedding (2 or 3 for visualization)
        - metric: str, distance metric; use 'precomputed' to pass a Gower distance
          matrix
        - random_state: int, seed for reproducibility
        - n_neighbors: int, size of the local neighborhood (UMAP's main locality
          parameter)
        - min_dist: float, minimum distance between points in the embedding
    Returns:
        - embedding: np.ndarray, embedded data of shape (n_samples, n_components)
    """
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return reducer.fit_transform(X)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Short labels so the method name fits inside each heatmap cell.
METHOD_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "cramers_v": "Cramér V",
    "point_biserial": "pt-bis",
    "correlation_ratio_eta": "eta",
    "identity": "",
    "insufficient_data": "n/a",
    "error": "err",
}


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    filename: str | None = None,
    method_matrix: pd.DataFrame | None = None,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] | None = None,
    annot: bool = True,
    triangle: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draws a heatmap of a mixed correlation/association matrix and optionally
    saves it to disk.
    Params:
        - corr_matrix: pd.DataFrame, square correlation matrix (index == columns)
        - title: str, plot title
        - filename: str | None, if given, the figure is saved to this path (dpi=300)
        - method_matrix: pd.DataFrame | None, matrix with the measure used for
          each pair; if given, it is annotated under each coefficient
        - cmap: str, colormap of the heatmap
        - figsize: tuple[float, float] | None, figure size; computed from the
          number of variables when None
        - annot: bool, if True annotate the numeric value of each cell
        - triangle: bool, if True show only the lower triangle (matrix is symmetric)
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the heatmap axes
    """
    n = corr_matrix.shape[0]

    if figsize is None:
        side = max(6, n * 0.95)
        figsize = (side + 1, side)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) if triangle else None

    # Construcción de las anotaciones (valor + método opcional).
    if annot and method_matrix is not None:
        labels = corr_matrix.copy().astype(object)
        for r in corr_matrix.index:
            for c in corr_matrix.columns:
                value = corr_matrix.loc[r, c]
                if np.isnan(value):
                    labels.loc[r, c] = ""
                else:
                    method = method_matrix.loc[r, c]
                    method_label = METHOD_LABELS.get(method, method)
                    labels.loc[r, c] = f"{value:.2f}\n{method_label}".rstrip()
        annot_data = labels.values
        fmt = ""
    else:
        annot_data = annot
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot_data,
        fmt=fmt,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Association / correlation"},
        annot_kws={"fontsize": 7},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_embedding(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray | list,
    title: str = "Embedding",
    axis_prefix: str = "Component",
    threeD: bool = False,
    filename: str | Path | None = None,
    palette: str = "Set1",
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter-plots a low-dimensional embedding (PCA, t-SNE, FAMD, UMAP, ...) colored
    by a label variable. Categorical labels (or numeric labels with few distinct
    values) get a discrete color per group with a legend; continuous numeric
    labels get a colorbar.
    Params:
        - embedding: np.ndarray, array of shape (n_samples, n_components); needs
          >= 3 columns if threeD else >= 2
        - labels: pd.Series | np.ndarray | list, values used to color the points
          (one per sample)
        - title: str, plot title
        - axis_prefix: str, prefix for the axis labels, e.g. "PC" -> "PC 1", "PC 2"
        - threeD: bool, if True draw a 3D scatter using the first three components
        - filename: str | Path | None, if given, save the figure (parent dirs are
          created)
        - palette: str, seaborn palette used for categorical labels
        - figsize: tuple[float, float], figure size
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the scatter axes (3D when threeD is True)
    """
    labels = pd.Series(labels)
    name = labels.name if labels.name is not None else "label"
    labels = labels.reset_index(drop=True)

    dims = 3 if threeD else 2
    if embedding.shape[1] < dims:
        raise ValueError(
            f"The embedding has {embedding.shape[1]} components but {dims} "
            "are required for this plot."
        )

    coords = [np.asarray(embedding)[:, i] for i in range(dims)]
    treat_as_continuous = pd.api.types.is_numeric_dtype(labels) and labels.nunique() > 10

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d") if threeD else fig.add_subplot(111)

    if treat_as_continuous:
        sc = ax.scatter(*coords, c=labels, cmap="viridis", s=60)
        fig.colorbar(sc, ax=ax, label=name, shrink=0.7)
    else:
        categories = pd.Categorical(labels.astype(str))
        colors = sns.color_palette(palette, len(categories.categories))
        for color, category in zip(colors, categories.categories):
            mask = categories == category
            ax.scatter(*[c[mask] for c in coords], color=[color], label=category, s=60)
        ax.legend(title=name)

    ax.set_title(title)
    ax.set_xlabel(f"{axis_prefix} 1")
    ax.set_ylabel(f"{axis_prefix} 2")
    if threeD:
        ax.set_zlabel(f"{axis_prefix} 3")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_pca(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray | list,
    title: str = "PCA Visualization",
    threeD: bool = False,
    filename: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper of plot_embedding for PCA projections (axis prefix "PC").
    Params:
        - embedding: np.ndarray, PCA coordinates of shape (n_samples, n_components)
        - labels: pd.Series | np.ndarray | list, values used to color the points
        - title: str, plot title
        - threeD: bool, if True draw a 3D scatter
        - filename: str | Path | None, if given, save the figure
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the scatter axes
    """
    return plot_embedding(
        embedding, labels, title=title, axis_prefix="PC", threeD=threeD, filename=filename
    )


def plot_tsne(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray | list,
    title: str = "t-SNE Visualization",
    threeD: bool = False,
    filename: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper of plot_embedding for t-SNE projections (axis prefix "t-SNE").
    Params:
        - embedding: np.ndarray, t-SNE coordinates of shape (n_samples, n_components)
        - labels: pd.Series | np.ndarray | list, values used to color the points
        - title: str, plot title
        - threeD: bool, if True draw a 3D scatter
        - filename: str | Path | None, if given, save the figure
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the scatter axes
    """
    return plot_embedding(
        embedding, labels, title=title, axis_prefix="t-SNE", threeD=threeD, filename=filename
    )


def plot_famd(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray | list,
    title: str = "FAMD Visualization",
    threeD: bool = False,
    filename: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper of plot_embedding for FAMD projections (axis prefix "FAMD").
    Params:
        - embedding: np.ndarray, FAMD coordinates of shape (n_samples, n_components)
        - labels: pd.Series | np.ndarray | list, values used to color the points
        - title: str, plot title
        - threeD: bool, if True draw a 3D scatter
        - filename: str | Path | None, if given, save the figure
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the scatter axes
    """
    return plot_embedding(
        embedding, labels, title=title, axis_prefix="FAMD", threeD=threeD, filename=filename
    )


def plot_umap(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray | list,
    title: str = "UMAP Visualization",
    threeD: bool = False,
    filename: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper of plot_embedding for UMAP projections (axis prefix "UMAP").
    Params:
        - embedding: np.ndarray, UMAP coordinates of shape (n_samples, n_components)
        - labels: pd.Series | np.ndarray | list, values used to color the points
        - title: str, plot title
        - threeD: bool, if True draw a 3D scatter
        - filename: str | Path | None, if given, save the figure
    Returns:
        - fig: matplotlib.figure.Figure, the created figure
        - ax: matplotlib.axes.Axes, the scatter axes
    """
    return plot_embedding(
        embedding, labels, title=title, axis_prefix="UMAP", threeD=threeD, filename=filename
    )

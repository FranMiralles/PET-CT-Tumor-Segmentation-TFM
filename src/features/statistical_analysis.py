from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, pointbiserialr, chi2_contingency
from scipy import stats



categorical_variables = ["Histopathological grading", "T-Stage", "N-Stage", "M-Stage", "Sex", "Smoking History"]
numerical_variables = ["weight (kg)", "Age"]



# CORRELATION ANALYSIS

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Computes Cramér's V for the association between two categorical variables.
    Params:
        - x: pd.Series, first categorical variable
        - y: pd.Series, second categorical variable
    Returns:
        - v: float, association in [0, 1] (0 = none, 1 = perfect); NaN if either
          variable has fewer than 2 distinct categories
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

def infer_variable_type(
    series_name: str,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> str:
    """
    Returns the variable type of a column based on the explicit column lists
    provided.
    Params:
        - series_name: str, name of the variable to classify
        - categorical_columns: list[str] | None, names of categorical variables
        - numerical_columns: list[str] | None, names of numeric variables
    Returns:
        - var_type: str, "categorical" or "numeric"; raises ValueError if the
          variable is not found in either list
    """
    categorical_columns = categorical_columns or []
    numerical_columns = numerical_columns or []

    if series_name in categorical_columns:
        return "categorical"
    if series_name in numerical_columns:
        return "numeric"

    raise ValueError(
        f"Cannot determine the type of '{series_name}': "
        "it must be listed in categorical_columns or numerical_columns."
    )


def correlation_ratio(
    categories: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
) -> float:
    """
    Computes the correlation ratio (eta) for the association between a
    categorical variable and a numeric variable.
    Params:
        - categories: pd.Series | np.ndarray, categorical/grouping variable
        - values: pd.Series | np.ndarray, numeric variable
    Returns:
        - eta: float, value in [0, 1] (0 = category explains none of the numeric
          variance, 1 = explains all of it); NaN if there are fewer than 2
          categories or the numeric variance is zero
    """
    data = pd.DataFrame({
        "categories": categories,
        "values": values
    }).dropna()

    if data["categories"].nunique() < 2:
        return np.nan

    grand_mean = data["values"].mean()

    ss_between = 0
    ss_total = ((data["values"] - grand_mean) ** 2).sum()

    for category, group in data.groupby("categories"):
        n_group = len(group)
        mean_group = group["values"].mean()
        ss_between += n_group * (mean_group - grand_mean) ** 2

    if ss_total == 0:
        return np.nan

    eta_squared = ss_between / ss_total

    return np.sqrt(eta_squared)

def _is_normal(normality_variables: dict, var: str) -> bool:
    """
    Reads a variable's normality flag in a robust way, accepting both the
    'is_normal' key (output of shapiro_wilk_test / dagostino_test) and
    'normality'. Assumes non-normal if no information is available.
    Params:
        - normality_variables: dict, {variable: {"is_normal"/"normality": bool, ...}}
        - var: str, variable whose normality flag is requested
    Returns:
        - is_normal: bool, True if the variable is flagged as normal, else False
    """
    info = normality_variables.get(var, {})
    if "is_normal" in info:
        return bool(info["is_normal"])
    return bool(info.get("normality", False))


def mixed_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    numeric_method: str = "spearman",
    normality_variables: dict | None = None,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    return_methods: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Computes an association/correlation matrix for mixed-type variables, choosing
    the measure according to each pair's types:
      - numeric vs numeric                     -> Pearson or Spearman
      - categorical vs categorical             -> Cramér's V
      - numeric vs categorical (2 categories)  -> point-biserial correlation
      - numeric vs categorical (>2 categories) -> correlation ratio (eta)
    The association measures lie in [0, 1] while Pearson/Spearman lie in [-1, 1];
    the resulting matrix is symmetric.
    Params:
        - df: pd.DataFrame, input data
        - columns: list[str] | None, columns to analyze; all columns if None
        - numeric_method: str, "pearson" or "spearman" for numeric-vs-numeric
          pairs when normality_variables is not provided
        - normality_variables: dict | None, {variable: {"is_normal": bool, ...}};
          if given, numeric pairs use Pearson only when both variables are normal
        - categorical_columns: list[str] | None, explicit categorical variable names
        - numerical_columns: list[str] | None, explicit numeric variable names
          (every analyzed column must appear in exactly one of these lists)
        - return_methods: bool, if True also return the method matrix and types
    Returns:
        - corr_matrix: pd.DataFrame, association/correlation matrix
        - method_matrix: pd.DataFrame, measure used for each pair (only if
          return_methods is True)
        - variable_types: dict, inferred type per variable (only if
          return_methods is True)
    """

    if numeric_method not in ("pearson", "spearman"):
        raise ValueError("numeric_method must be 'pearson' or 'spearman'.")

    if columns is None:
        columns = df.columns.tolist()

    data = df[columns].copy()

    variable_types = {
        col: infer_variable_type(col, categorical_columns, numerical_columns)
        for col in columns
    }

    corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
    method_matrix = pd.DataFrame("", index=columns, columns=columns, dtype=object)

    for col1, col2 in combinations(columns, 2):
        x_clean, y_clean = data[col1], data[col2]
        pair = pd.DataFrame({"x": x_clean, "y": y_clean}).dropna()

        if len(pair) < 2:
            value, method = np.nan, "insufficient_data"
        else:
            x_clean = pair["x"]
            y_clean = pair["y"]
            type_x = variable_types[col1]
            type_y = variable_types[col2]

            try:
                # Numeric vs numeric
                if type_x == "numeric" and type_y == "numeric":
                    if normality_variables is not None:
                        both_normal = _is_normal(normality_variables, col1) and _is_normal(normality_variables, col2)
                        method = "pearson" if both_normal else "spearman"
                    else:
                        method = numeric_method

                    if method == "pearson":
                        value, _ = pearsonr(x_clean, y_clean)
                    else:
                        value, _ = spearmanr(x_clean, y_clean)

                # Categorical vs categorical
                elif type_x == "categorical" and type_y == "categorical":
                    value = cramers_v(x_clean, y_clean)
                    method = "cramers_v"

                # Numeric vs categorical
                else:
                    if type_x == "numeric":
                        numeric, categorical = x_clean, y_clean
                    else:
                        numeric, categorical = y_clean, x_clean

                    if categorical.nunique() == 2:
                        codes = pd.Categorical(categorical).codes
                        value, _ = pointbiserialr(codes, numeric)
                        method = "point_biserial"
                    else:
                        value = correlation_ratio(categorical, numeric)
                        method = "correlation_ratio_eta"

            except Exception:
                value, method = np.nan, "error"

        # The matrix is symmetric: fill both sides.
        corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = value
        method_matrix.loc[col1, col2] = method_matrix.loc[col2, col1] = method

    # Diagonal.
    for col in columns:
        corr_matrix.loc[col, col] = 1.0
        method_matrix.loc[col, col] = "identity"

    if return_methods:
        return corr_matrix, method_matrix, variable_types

    return corr_matrix



# GAUSSIAN ANALYSIS

def shapiro_wilk_test(df: pd.DataFrame) -> dict:
    '''
    Applies the Shapiro-Wilk normality test column by column. NaN values are
    dropped before testing each variable.
    Params:
        - df: pd.DataFrame, DataFrame whose columns are the variables to test
    Returns:
        - results: dict, {variable: {"statistic": float, "p_value": float,
          "is_normal": bool}}, where "is_normal" is True when p_value > 0.05
    '''
    results = {}
    for var in df.columns:
        datos = df[var].dropna()
        # Shapiro-Wilk analysis
        shapiro_stat, shapiro_p = stats.shapiro(datos)
        results[var] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
    return results

def dagostino_test(df: pd.DataFrame) -> dict:
    '''
    Applies the D'Agostino-Pearson K-squared omnibus normality test column by
    column (combines skewness and kurtosis). NaN values are dropped before
    testing each variable.
    Params:
        - df: pd.DataFrame, DataFrame whose columns are the variables to test
    Returns:
        - results: dict, {variable: {"statistic": float, "p_value": float,
          "is_normal": bool}}, where "is_normal" is True when p_value > 0.05
    '''
    results = {}
    for var in df.columns:
        datos = df[var].dropna()
        # D'Agostino's test
        dagostino_stat, dagostino_p = stats.normaltest(datos)
        results[var] = {
            'statistic': dagostino_stat,
            'p_value': dagostino_p,
            'is_normal': dagostino_p > 0.05
        }
    return results



# UNIVARIATE ANALYSIS

def univariate_analysis(
    df: pd.DataFrame,
    column: str,
    normality_variables: dict,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Runs a univariate analysis between a target variable and every other variable
    in the DataFrame, choosing the appropriate statistical test from the pair of
    types and (for normality-dependent tests) the numeric variables' normality:
      - categorical vs categorical          -> Chi-square test of independence
      - numeric vs categorical (normal)     -> Welch t-test (2 groups) / ANOVA (>2)
      - numeric vs categorical (non-normal) -> Mann-Whitney U (2) / Kruskal-Wallis (>2)
      - numeric vs numeric                  -> Pearson (both normal) / Spearman
    Params:
        - df: pd.DataFrame, input data; all columns except `column` must be listed
          in categorical_columns or numerical_columns
        - column: str, target/study variable, compared against all other columns
        - normality_variables: dict, {variable: {"is_normal": bool, ...}} as
          returned by shapiro_wilk_test / dagostino_test
        - categorical_columns: list[str] | None, explicit categorical variable names
        - numerical_columns: list[str] | None, explicit numeric variable names
    Returns:
        - results_df: pd.DataFrame, one row per variable sorted by ascending
          p_value, with columns variable, variable_type, study_variable, test,
          statistic, p_value and p_value_str
    """
    results = []
    y_type = infer_variable_type(column, categorical_columns, numerical_columns)

    for var in df.columns.drop(column):
        x_type = infer_variable_type(var, categorical_columns, numerical_columns)

        data = df[[var, column]].dropna()
        x_clean = data[var]
        y_clean = data[column]

        # Categorical vs categorical
        if x_type == "categorical" and y_type == "categorical":
            table = pd.crosstab(x_clean, y_clean)
            stat, p, _, _ = stats.chi2_contingency(table)
            test = "Chi-square"

        # Numeric vs numeric
        elif x_type == "numeric" and y_type == "numeric":
            both_normal = (
                _is_normal(normality_variables, var)
                and _is_normal(normality_variables, column)
            )
            if both_normal:
                stat, p = stats.pearsonr(x_clean, y_clean)
                test = "Pearson"
            else:
                stat, p = stats.spearmanr(x_clean, y_clean)
                test = "Spearman"

        # Numeric vs categorical: compare the numeric distribution across the
        # groups defined by the categorical variable.
        else:
            if x_type == "numeric":
                numeric, grouping, numeric_name = x_clean, y_clean, var
            else:
                numeric, grouping, numeric_name = y_clean, x_clean, column

            groups = [numeric[grouping == g] for g in grouping.unique()]

            if _is_normal(normality_variables, numeric_name):
                if len(groups) == 2:
                    stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test = "Welch t-test"
                else:
                    stat, p = stats.f_oneway(*groups)
                    test = "ANOVA"
            else:
                if len(groups) == 2:
                    stat, p = stats.mannwhitneyu(groups[0], groups[1])
                    test = "Mann-Whitney U"
                else:
                    stat, p = stats.kruskal(*groups)
                    test = "Kruskal-Wallis"

        results.append({
            "variable": var,
            "variable_type": x_type,
            "study_variable": column,
            "test": test,
            "statistic": stat,
            "p_value": p,
        })

    results_df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    results_df["p_value_str"] = results_df["p_value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    return results_df

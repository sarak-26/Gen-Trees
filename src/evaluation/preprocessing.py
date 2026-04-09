import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def strip_object_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    return df


def infer_types_by_numeric_coercion(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_threshold: float = 0.95,
    categorical_cols=None,
    numeric_cols=None,
):
    shared = sorted(set(real_df.columns).intersection(set(synth_df.columns)))
    if not shared:
        raise ValueError("real.csv and synthetic.csv share no columns.")
    real_df = real_df[shared].copy()
    synth_df = synth_df[shared].copy()

    real_df = strip_object_whitespace(real_df)
    synth_df = strip_object_whitespace(synth_df)

    if categorical_cols is not None or numeric_cols is not None:
        if categorical_cols is None:
            categorical_cols = [c for c in shared if c not in (numeric_cols or [])]
        if numeric_cols is None:
            numeric_cols = [c for c in shared if c not in (categorical_cols or [])]
        return real_df, synth_df, categorical_cols, numeric_cols

    categorical_cols = []
    numeric_cols = []

    for c in shared:
        r_num = pd.to_numeric(real_df[c], errors="coerce")
        s_num = pd.to_numeric(synth_df[c], errors="coerce")

        r_ok = float(r_num.notna().mean())
        s_ok = float(s_num.notna().mean())

        if r_ok >= numeric_threshold and s_ok >= numeric_threshold:
            numeric_cols.append(c)
            real_df[c] = r_num
            synth_df[c] = s_num
        else:
            categorical_cols.append(c)

    return real_df, synth_df, categorical_cols, numeric_cols


def remove_id_like_and_constant(real_df, synth_df, numeric_cols, categorical_cols, id_unique_threshold=0.98):
    to_drop = []

    def nunique_ratio(s: pd.Series) -> float:
        s2 = s.dropna()
        if len(s2) == 0:
            return 0.0
        return float(s2.nunique()) / float(len(s2))

    for c in numeric_cols + categorical_cols:
        r = nunique_ratio(real_df[c])
        s = nunique_ratio(synth_df[c])

        if r >= id_unique_threshold and s >= id_unique_threshold:
            to_drop.append(c)
            continue
        if real_df[c].dropna().nunique() <= 1 and synth_df[c].dropna().nunique() <= 1:
            to_drop.append(c)

    if to_drop:
        real_df = real_df.drop(columns=to_drop, errors="ignore")
        synth_df = synth_df.drop(columns=to_drop, errors="ignore")
        numeric_cols = [c for c in numeric_cols if c not in to_drop]
        categorical_cols = [c for c in categorical_cols if c not in to_drop]

    return real_df, synth_df, numeric_cols, categorical_cols, to_drop


def split_numeric_by_cardinality(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols,
    discrete_cardinality_threshold: int = 7,
):
    one_hot_numeric_cols = []
    continuous_numeric_cols = []

    for c in numeric_cols or []:
        combined = pd.concat([real_df[c], synth_df[c]], axis=0, ignore_index=True)
        cardinality = int(combined.dropna().nunique())
        if cardinality <= discrete_cardinality_threshold:
            one_hot_numeric_cols.append(c)
        else:
            continuous_numeric_cols.append(c)

    return one_hot_numeric_cols, continuous_numeric_cols


def one_hot_encode_aligned(
    real_df,
    synth_df,
    categorical_cols,
    numeric_cols,
    discrete_cardinality_threshold: int = 7,
):
    one_hot_numeric_cols, continuous_numeric_cols = split_numeric_by_cardinality(
        real_df,
        synth_df,
        numeric_cols=numeric_cols,
        discrete_cardinality_threshold=discrete_cardinality_threshold,
    )
    one_hot_cols = list(categorical_cols or []) + one_hot_numeric_cols
    combined = pd.concat(
        [synth_df[one_hot_cols + continuous_numeric_cols], real_df[one_hot_cols + continuous_numeric_cols]],
        axis=0,
        ignore_index=True,
    )
    X = pd.get_dummies(combined, columns=one_hot_cols, dummy_na=True)

    X_synth = X.iloc[: len(synth_df)].to_numpy(dtype=float)
    X_real = X.iloc[len(synth_df):].to_numpy(dtype=float)
    return X_real, X_synth, X.columns.tolist()


def build_global_utility_preprocessor(
    feature_cols,
    categorical_cols,
    numeric_cols,
    real_df: pd.DataFrame | None = None,
    synth_df: pd.DataFrame | None = None,
    discrete_cardinality_threshold: int = 7,
):
    one_hot_numeric_cols = []
    if real_df is not None and synth_df is not None:
        one_hot_numeric_cols, _ = split_numeric_by_cardinality(
            real_df,
            synth_df,
            numeric_cols=numeric_cols,
            discrete_cardinality_threshold=discrete_cardinality_threshold,
        )

    cat = [c for c in feature_cols if c in ((categorical_cols or []) + one_hot_numeric_cols)]
    num = [c for c in feature_cols if c in (numeric_cols or []) and c not in one_hot_numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(with_mean=True, with_std=True), num),
        ],
        remainder="drop",
    )

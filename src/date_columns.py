from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from dateutil import parser as date_parser


KNOWN_DATE_FORMATS = (
    "%d-%b-%y",
    "%d-%b-%Y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
)

DATE_NAME_TOKENS = ("date", "time", "timestamp", "dob")


@dataclass
class DateColumnMetadata:
    original_format: str | None
    min_timestamp: int | None
    max_timestamp: int | None
    parse_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_format": self.original_format,
            "min_timestamp": self.min_timestamp,
            "max_timestamp": self.max_timestamp,
            "parse_strategy": self.parse_strategy,
        }


def _looks_like_date_name(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return any(token in normalized for token in DATE_NAME_TOKENS)


def _infer_format_from_values(series: pd.Series) -> str | None:
    sample = series.dropna().astype(str).str.strip()
    if sample.empty:
        return None

    values = sample[sample.ne("")].head(100).tolist()
    if not values:
        return None

    for candidate in KNOWN_DATE_FORMATS:
        try:
            pd.to_datetime(values, format=candidate, errors="raise")
            return candidate
        except (TypeError, ValueError):
            continue
    return None


def _row_wise_parse(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    try:
        return pd.Timestamp(date_parser.parse(text))
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def _parse_date_series(series: pd.Series) -> tuple[pd.Series, str] | tuple[None, None]:
    non_null = series.dropna()
    if non_null.empty:
        return None, None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format' is deprecated")
        warnings.filterwarnings("ignore", message="Could not infer format")
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    if parsed.loc[non_null.index].notna().all():
        return parsed, "infer_datetime_format"

    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if parsed.loc[non_null.index].notna().all():
        return parsed, "dayfirst"

    parsed = series.map(_row_wise_parse)
    if parsed.loc[non_null.index].notna().all():
        return pd.to_datetime(parsed, errors="coerce"), "dateutil_rowwise"

    return None, None


def _parse_date_series_with_metadata(
    series: pd.Series,
    metadata: dict[str, Any],
) -> pd.Series:
    original_format = metadata.get("original_format")
    if original_format:
        parsed = pd.to_datetime(series, format=original_format, errors="coerce")
        non_null = series.dropna()
        if non_null.empty or parsed.loc[non_null.index].notna().all():
            return parsed

    parsed, _ = _parse_date_series(series)
    if parsed is None:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return parsed


def _datetime_to_unix_seconds(series: pd.Series) -> pd.Series:
    encoded = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = series.notna()
    if valid.any():
        encoded.loc[valid] = (series.loc[valid].astype("int64") // 10**9).astype("float64")
    return encoded


def detect_and_encode_date_columns(
    df: pd.DataFrame,
    *,
    candidate_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    encoded_df = df.copy()
    date_metadata: dict[str, dict[str, Any]] = {}

    if candidate_columns is None:
        candidate_columns = [
            column
            for column in encoded_df.columns
            if _looks_like_date_name(column) or _infer_format_from_values(encoded_df[column]) is not None
        ]

    for column in candidate_columns:
        if column not in encoded_df.columns:
            continue

        parsed, parse_strategy = _parse_date_series(encoded_df[column])
        if parsed is None:
            continue

        timestamps = _datetime_to_unix_seconds(parsed)
        valid = timestamps.dropna()
        if valid.empty:
            continue

        encoded_df[column] = timestamps
        date_metadata[column] = DateColumnMetadata(
            original_format=_infer_format_from_values(df[column]),
            min_timestamp=int(valid.min()),
            max_timestamp=int(valid.max()),
            parse_strategy=parse_strategy,
        ).to_dict()

    return encoded_df, date_metadata


def apply_date_metadata_and_encode(
    df: pd.DataFrame,
    date_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    encoded_df = df.copy()
    for column, metadata in date_metadata.items():
        if column not in encoded_df.columns:
            continue
        parsed = _parse_date_series_with_metadata(encoded_df[column], metadata)
        encoded_df[column] = _datetime_to_unix_seconds(parsed)
    return encoded_df


def decode_date_columns(
    df: pd.DataFrame,
    date_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    decoded_df = df.copy()
    for column, metadata in date_metadata.items():
        if column not in decoded_df.columns:
            continue

        numeric = pd.to_numeric(decoded_df[column], errors="coerce")
        lower = metadata.get("min_timestamp")
        upper = metadata.get("max_timestamp")
        if lower is not None and upper is not None:
            numeric = numeric.clip(lower=lower, upper=upper)

        decoded = pd.to_datetime(numeric, unit="s", errors="coerce")
        original_format = metadata.get("original_format") or "%Y-%m-%d"
        formatted = decoded.dt.strftime(original_format)
        decoded_df[column] = formatted.where(decoded.notna(), pd.NA)

    return decoded_df


def prepare_training_dates(df: pd.DataFrame) -> pd.DataFrame:
    encoded_df, date_metadata = detect_and_encode_date_columns(df)
    encoded_df.attrs["date_column_metadata"] = date_metadata
    return encoded_df


def finalize_synthetic_dates(df: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    finalized = df.reindex(columns=training_df.columns)
    date_metadata = training_df.attrs.get("date_column_metadata", {})
    if date_metadata:
        finalized = decode_date_columns(finalized, date_metadata)
    return finalized

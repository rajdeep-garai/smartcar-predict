import re
from typing import Tuple, List
import pandas as pd
import numpy as np


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df: pd.DataFrame, keywords: Tuple[str, ...]) -> str | None:
    for col in df.columns:
        name = normalize_column_name(str(col))
        if any(normalize_column_name(k) in name for k in keywords):
            return str(col)
    return None


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^\d.\-]", "", regex=True)
    cleaned = cleaned.replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def identify_features(df: pd.DataFrame, target: str) -> Tuple[List[str], dict]:
    feature_rules = (
        ("brand", ("brand", "make")),
        ("model", ("model", "variant", "car_name")),
        ("year", ("year", "vehicle_age")),
        ("kilometers", ("km_driven", "kilometer", "odometer", "mileage_km")),
        ("fuel", ("fuel", "fuel_type")),
        ("transmission", ("transmission", "gear")),
        ("owner", ("owner", "seller_type")),
        ("mileage", ("mileage",)),
        ("engine", ("engine", "engine_capacity", "displacement")),
        ("power", ("max_power", "power", "bhp")),
        ("seats", ("seats",)),
    )
    features: List[str] = []
    column_map = {}
    for logical, keywords in feature_rules:
        col = find_column(df, keywords)
        if col and col != target and col not in features:
            if logical == "model" and col.lower() == "car_name" and "model" in [f.lower() for f in features]:
                continue
            features.append(col)
            column_map[logical] = col
    if not features:
        features = [c for c in df.columns if c != target]
    return features[:30], column_map


def add_derived_features(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    result = df.copy()
    valuation_year = find_column(result, ("valuationyear", "sellingyear", "year"))
    age_col = find_column(result, ("vehicleage", "age"))
    km_col = column_map.get("kilometers") or find_column(result, ("km_driven", "kilometer", "odometer"))

    if valuation_year and age_col:
        # Vehicle age is used directly when no manufacturing year is present.
        pass
    elif age_col:
        pass

    manufacturing_col = find_column(result, ("manufacturingyear", "yearofmanufacture", "make_year"))
    if manufacturing_col:
        manufacturing_col = str(manufacturing_col)
        result[manufacturing_col] = pd.to_numeric(result[manufacturing_col], errors="coerce")
        reference_year = pd.Timestamp.now().year
        derived = np.maximum(reference_year - result[manufacturing_col], 1)
        result["_derived_car_age"] = derived
        if km_col:
            km_col = str(km_col)
            result["_derived_usage_per_year"] = result[km_col].fillna(0) / np.maximum(derived, 1)
    elif age_col:
        age_col = str(age_col)
        numeric_age = pd.to_numeric(result[age_col], errors="coerce").clip(lower=1)
        if km_col:
            km_col = str(km_col)
            result["_derived_usage_per_year"] = result[km_col].fillna(0) / np.maximum(numeric_age, 1)
    return result


def clean_and_engineer(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    data = df.drop_duplicates().copy()
    data[target] = clean_numeric_series(data[target])
    data = data[data[target].notna() & (data[target] > 0)]

    name_col = find_column(data, ("name", "car_name"))
    if name_col:
        name_text = data[name_col].fillna("").astype(str).str.strip()
        data["_derived_brand"] = name_text.str.extract(r"^([A-Za-z0-9]+)", expand=False)
        data["_derived_model"] = name_text.str.replace(
            r"^[A-Za-z0-9]+\s+", "", regex=True
        ).str.strip()

    possible_numeric = (
        "year", "age", "km", "kilometer", "odometer", "mileage",
        "engine", "power", "bhp", "seat", "cc", "capacity", "price"
    )
    for col in data.columns:
        if col == target or any(k in normalize_column_name(str(col)) for k in possible_numeric):
            data[col] = clean_numeric_series(data[col])

    features, column_map = identify_features(data, target)
    data = add_derived_features(data, column_map)

    derived_features = [c for c in data.columns if c.startswith("_derived_") and c not in features]
    features.extend(derived_features)

    # Sensible filters for impossible and extreme values.
    impossible_rules = (
        ("year", (1880, 2030)),
        ("age", (0, 100)),
        ("km", (0, 1_500_000)),
        ("kilometer", (0, 1_500_000)),
        ("odometer", (0, 1_500_000)),
        ("seat", (1, 15)),
    )
    for col in features:
        name = normalize_column_name(str(col))
        for keyword, (low, high) in impossible_rules:
            if keyword in name:
                data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=low, upper=high)
                break

    # Filter extreme target values with IQR while preserving a sensible sample.
    if data[target].count() >= 20:
        q1, q3 = data[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            data = data[data[target].between(low, high)]

    features = [c for c in features if c in data.columns]
    data = data.dropna(subset=[target]).copy()
    if len(data) < 20:
        raise ValueError("Insufficient valid records for training. At least 20 records are required.")
    return data, {
        "features": features,
        "target": target,
        "column_map": column_map,
    }

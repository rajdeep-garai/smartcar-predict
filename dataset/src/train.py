from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import shap

from src.data_loader import load_dataset, find_target_column, PROJECT_ROOT
from src.preprocessing import clean_and_engineer
from src.evaluate import calculate_metrics


def build_pipeline(features: list, model):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(model._smartcar_data[c])]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", model),
    ])


def build_depreciation_metadata(data: pd.DataFrame, target: str) -> dict:
    def normalized(value):
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    brand_col = next((c for c in data.columns if "brand" in normalized(c) or "make" in normalized(c)), None)
    model_col = next((c for c in data.columns if "carname" in normalized(c)), None)
    if model_col is None:
        model_col = next((c for c in data.columns if "model" in normalized(c) and c != target), None)
    age_col = next((c for c in data.columns if "vehicleage" in normalized(c) or "age" in normalized(c)), None)
    km_col = next((c for c in data.columns if "km" in normalized(c) or "odometer" in normalized(c) or "kilometer" in normalized(c)), None)
    if not age_col and not km_col:
        return {"available": False}

    def isotonic_curve(frame, value_col):
        frame = frame[[value_col, target]].copy()
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.dropna()
        if len(frame) < 5:
            return None

        grouped = frame.groupby(value_col)[target].median()
        if len(grouped) < 2:
            return None
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float))
        anchor = float(iso.predict([float(grouped.index.min())])[0])
        if anchor <= 0:
            return None
        return {
            "values": [float(v) for v in grouped.index],
            "prices": [float(v) for v in iso.predict(grouped.index.to_numpy(dtype=float))],
            "anchor": anchor,
        }

    metadata = {
        "available": True,
        "age_column": age_col,
        "km_column": km_col,
        "brand_column": brand_col,
        "model_column": model_col,
        "model_profiles": {},
        "exact_age": {},
        "brand_age": {},
        "exact_km": {},
        "brand_km": {},
    }

    if model_col:
        for model_name, frame in data.groupby(model_col, dropna=True):
            if len(frame) < 5:
                continue
            profile_frame = frame[[age_col, km_col, target]].copy() if age_col and km_col else frame[[target]].copy()
            profile_frame = profile_frame.dropna()
            if profile_frame.empty or profile_frame[target].count() == 0:
                continue
            profile = {
                "overall_median": float(profile_frame[target].median()),
                "median_km": float(pd.to_numeric(profile_frame[km_col], errors="coerce").median()) if km_col else None,
                "age_medians": {},
                "age_km_medians": {},
            }
            if age_col:
                for age, age_frame in profile_frame.groupby(age_col):
                    profile["age_medians"][str(int(age))] = float(age_frame[target].median())
                    profile["age_km_medians"][str(int(age))] = float(pd.to_numeric(age_frame[km_col], errors="coerce").median()) if km_col else None
            metadata["model_profiles"][str(model_name)] = profile
            if age_col:
                curve = isotonic_curve(frame, age_col)
                if curve:
                    metadata["exact_age"][str(model_name)] = curve
            if km_col:
                km_frame = frame[[km_col, target]].copy()
                km_frame["_km_band"] = (pd.to_numeric(km_frame[km_col], errors="coerce") // 10_000 * 10_000)
                curve = isotonic_curve(km_frame.rename(columns={"_km_band": "_km_band"})[["_km_band", target]], "_km_band")
                if curve:
                    metadata["exact_km"][str(model_name)] = curve

    if brand_col:
        for brand_name, frame in data.groupby(brand_col, dropna=True):
            if len(frame) < 10:
                continue
            if age_col:
                curve = isotonic_curve(frame, age_col)
                if curve:
                    metadata["brand_age"][str(brand_name)] = curve
            if km_col:
                km_frame = frame[[km_col, target]].copy()
                km_frame["_km_band"] = (pd.to_numeric(km_frame[km_col], errors="coerce") // 10_000 * 10_000)
                curve = isotonic_curve(km_frame.rename(columns={"_km_band": "_km_band"})[["_km_band", target]], "_km_band")
                if curve:
                    metadata["brand_km"][str(brand_name)] = curve

    return metadata


def train_models(data: pd.DataFrame, features: list, target: str):
    X = data[features].copy()
    y = data[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers)
    candidates = {
        "CatBoost": Pipeline([("preprocessor", preprocessor),
                              ("model", CatBoostRegressor(
                                  iterations=500,
                                  learning_rate=0.08,
                                  depth=8,
                                  loss_function="RMSE",
                                  random_seed=42,
                                  verbose=False,
                                  thread_count=-1,
                              ))]),
        "Random Forest": Pipeline([("preprocessor", preprocessor),
                                   ("model", RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1))]),
        "XGBoost": Pipeline([("preprocessor", preprocessor),
                             ("model", XGBRegressor(
                                 n_estimators=300,
                                 learning_rate=0.08,
                                 max_depth=7,
                                 subsample=0.9,
                                 colsample_bytree=0.9,
                                 objective="reg:squarederror",
                                 random_state=42,
                                 n_jobs=-1,
                             ))]),
    }

    model_results = {}
    for name, pipeline in candidates.items():
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
            validation_mae = float(-scores.mean())
            pipeline.fit(X_train, y_train)
            test_pred = pipeline.predict(X_test)
            test_metrics = calculate_metrics(y_test, test_pred)
            model_results[name] = {
                "pipeline": pipeline,
                "validation_mae": validation_mae,
                "test_metrics": test_metrics,
                "predictions": test_pred,
                "y_test": y_test,
            }
        except Exception as exc:
            model_results[name] = {"error": str(exc)}

    successful = {k: v for k, v in model_results.items() if "error" not in v}
    if not successful:
        raise RuntimeError("Both model pipelines failed to train.")
    selected_name = "CatBoost" if "CatBoost" in successful else min(successful, key=lambda k: successful[k]["validation_mae"])
    selected = successful[selected_name]
    try:
        sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
        selected_pipeline = selected["pipeline"]
        transformed = selected_pipeline.named_steps["preprocessor"].transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = shap.TreeExplainer(selected_pipeline.named_steps["model"])
        shap_values = explainer.shap_values(transformed)
        shap_values = np.asarray(shap_values)
        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_names = selected_pipeline.named_steps["preprocessor"].get_feature_names_out()
        shap_summary = {
            str(name): float(value)
            for name, value in sorted(
                zip(feature_names, mean_abs),
                key=lambda item: item[1],
                reverse=True,
            )[:15]
        }
    except Exception:
        shap_summary = {}
    metadata = {
        "selected_model": selected_name,
        "target": target,
        "features": features,
        "depreciation": build_depreciation_metadata(data, target),
        "categories": {
            c: sorted([str(v) for v in data[c].dropna().unique().tolist()])[:100]
            for c in categorical_features
        },
        "numeric_ranges": {
            c: {
                "min": float(pd.to_numeric(data[c], errors="coerce").min()),
                "max": float(pd.to_numeric(data[c], errors="coerce").max()),
                "median": float(pd.to_numeric(data[c], errors="coerce").median()),
            }
            for c in numeric_features
        },
        "model_metrics": {name: value["test_metrics"] for name, value in model_results.items()
                          if "test_metrics" in value},
        "validation_mae": {name: value["validation_mae"] for name, value in successful.items()},
        "test_metrics": selected["test_metrics"],
        "test_predictions": [float(v) for v in selected["predictions"]],
        "test_actual": [float(v) for v in selected["y_test"].tolist()],
        "shap_summary": shap_summary,
    }
    return selected["pipeline"], metadata, model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "dataset" / "car_data.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "car_price_model.pkl"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    target = find_target_column(df)
    if not target:
        raise ValueError("Could not identify a target price column.")
    data, info = clean_and_engineer(df, target)
    model, metadata, results = train_models(data, info["features"], target)
    processed_path = PROJECT_ROOT / "dataset" / "processed_data.csv"
    data.to_csv(processed_path, index=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "metadata": metadata}, output)
    print(f"Dataset shape: {df.shape}")
    print(f"Training records: {len(data)}")
    print(f"Target: {target}")
    print(f"Features: {info['features']}")
    print("Validation MAE:")
    for name, mae in metadata["validation_mae"].items():
        print(f"  {name}: {mae:,.2f}")
    print(f"Selected model: {metadata['selected_model']}")
    print("Test metrics:")
    for name, values in metadata["test_metrics"].items():
        print(f"  {name}: {values:,.2f}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()

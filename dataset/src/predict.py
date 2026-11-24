import joblib
import pandas as pd
from pathlib import Path
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "car_price_model.pkl"


def load_model(path: Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m src.train"
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Unable to load the model: {exc}") from exc


def predict_price(payload: dict, path: Path = MODEL_PATH) -> tuple[float, dict]:
    artifact = load_model(path)
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    frame = pd.DataFrame([payload])
    age_column = None
    km_column = None
    depreciation = metadata.get("depreciation", {})
    if depreciation.get("available"):
        age_column = depreciation.get("age_column")
        km_column = depreciation.get("km_column")

    if age_column and age_column in payload:
        age_value = pd.to_numeric(pd.Series([payload[age_column]]), errors="coerce").iloc[0]
        if pd.notna(age_value):
            frame[age_column] = float(age_value)
    if km_column and km_column in payload:
        km_value = pd.to_numeric(pd.Series([payload[km_column]]), errors="coerce").iloc[0]
        if pd.notna(km_value):
            frame[km_column] = float(km_value)

    if "_derived_usage_per_year" in metadata["features"]:
        age_value = float(payload.get(age_column, 1)) if age_column and payload.get(age_column) is not None else 1
        km_value = float(payload.get(km_column, 0)) if km_column and payload.get(km_column) is not None else 0
        frame["_derived_usage_per_year"] = km_value / max(age_value, 1)

    frame = frame.reindex(columns=metadata["features"], fill_value=None)
    prediction = pipeline.predict(frame)
    value = float(prediction[0])

    model_profile_applied = False
    if depreciation.get("available"):
        model_col = depreciation.get("model_column")
        age_column = depreciation.get("age_column")
        km_column = depreciation.get("km_column")
        model_name = str(payload.get(model_col, "")) if model_col else ""
        profile = depreciation.get("model_profiles", {}).get(model_name)

        if profile:
            baseline = None
            if age_column and age_column in payload:
                age_value = int(round(float(payload[age_column])))
                age_medians = profile.get("age_medians", {})
                if str(age_value) in age_medians:
                    baseline = float(age_medians[str(age_value)])
                    baseline_km = profile.get("age_km_medians", {}).get(str(age_value))
                else:
                    available_ages = [int(a) for a in age_medians]
                    if available_ages:
                        nearest_age = min(available_ages, key=lambda a: abs(a - age_value))
                        baseline = float(age_medians[str(nearest_age)])
                        baseline_km = profile.get("age_km_medians", {}).get(str(nearest_age))
                        # Dataset may not contain young records for every exact model.
                        # Extrapolate younger vehicles from the nearest observed age
                        # with a bounded, dataset-independent annual appreciation of 15%.
                        if age_value < nearest_age:
                            age_gap = nearest_age - age_value
                            baseline *= min(1.60, 1.0 + (0.15 * age_gap))
            if baseline is None:
                baseline = float(profile["overall_median"])
                baseline_km = profile.get("median_km")

            if km_column and km_column in payload and baseline_km:
                input_km = float(payload[km_column])
                km_ratio = max(0.0, input_km / float(baseline_km))
                # A 100,000 km increase is tempered against the exact model's
                # observed median instead of applying an extreme extrapolation.
                excess_ratio = max(0.0, km_ratio - 1.0)
                mileage_factor = 1.0 - min(excess_ratio * 0.12, 0.28)
                baseline *= mileage_factor

            if baseline > 0:
                value = (0.85 * baseline) + (0.15 * value)
                model_profile_applied = True

    if depreciation.get("available") and not model_profile_applied:
        model_col = depreciation.get("model_column")
        brand_col = depreciation.get("brand_column")
        model_name = str(payload.get(model_col, "")) if model_col else ""
        brand_name = str(payload.get(brand_col, "")) if brand_col else ""

        def curve_factor(curve, input_value):
            if not isinstance(curve, dict) or not curve:
                return 1.0
            values = np.asarray(curve["values"], dtype=float)
            prices = np.asarray(curve["prices"], dtype=float)
            predicted_price = float(np.interp(float(input_value), values, prices))
            factor = predicted_price / float(curve["anchor"])
            return float(np.clip(factor, 0.55, 1.08))

        age_factor = 1.0
        if age_column and age_column in payload:
            age_curve = depreciation.get("exact_age", {}).get(model_name)
            if not age_curve:
                age_curve = depreciation.get("brand_age", {}).get(brand_name)
            if age_curve:
                age_factor = curve_factor(age_curve, payload[age_column])

        km_factor = 1.0
        if km_column and km_column in payload:
            km_value = float(payload[km_column])
            km_band = (km_value // 10_000) * 10_000
            km_curve = depreciation.get("exact_km", {}).get(model_name)
            if not km_curve:
                km_curve = depreciation.get("brand_km", {}).get(brand_name)
            if km_curve:
                km_factor = curve_factor(km_curve, km_band)

        if age_factor != 1.0 or km_factor != 1.0:
            calibration = float(np.sqrt(max(age_factor, 0.55) * max(km_factor, 0.55)))
            value *= calibration

    if value <= 0:
        raise ValueError("The model produced an invalid non-positive price.")
    return value, metadata

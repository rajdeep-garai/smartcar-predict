from pathlib import Path
from typing import Optional
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "car_data.csv"


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    """Find a likely Indian used-car price target column."""
    normalized = {col: "".join(ch for ch in str(col).lower() if ch.isalnum()) for col in df.columns}
    candidates = ("sellingprice", "saleprice", "price", "resaleprice", "marketprice")
    for key, normalized_name in normalized.items():
        if normalized_name in candidates:
            return str(key)
    for key, normalized_name in normalized.items():
        if "price" in normalized_name or "cost" in normalized_name:
            return str(key)
    return None


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle dataset or add a CSV named car_data.csv."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    target = find_target_column(df)
    return {
        "shape": df.shape,
        "head": df.head(),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes,
        "missing": df.isna().sum(),
        "duplicates": int(df.duplicated().sum()),
        "target": target,
    }

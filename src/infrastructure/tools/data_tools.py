import pandas as pd
import numpy as np
from langchain_core.tools import tool


@tool
def calculate_statistics(data: list[float]) -> dict:
    """Calculate statistical measures for a list of numbers.

    Args:
        data: List of numeric values

    Returns:
        Dictionary with mean, median, std_dev, min, max
    """
    try:
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(data),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def analyze_csv(csv_path: str) -> dict:
    """Analyze a CSV file and return basic statistics.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with column statistics
    """
    try:
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        result = {
            "rows": len(df),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "statistics": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        }
        return result
    except Exception as e:
        return {"error": str(e)}

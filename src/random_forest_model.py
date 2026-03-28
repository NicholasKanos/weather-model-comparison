import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_random_forest(file_path: str):
    df = pd.read_csv(file_path)

    # Convert and sort date
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")

    # Create target (next day max temp)
    df["TARGET"] = df["TMAX"].shift(-1)

    # Create lag features
    for lag in range(1, 15):
        df[f"TMAX_lag{lag}"] = df["TMAX"].shift(lag)
        df[f"TMIN_lag{lag}"] = df["TMIN"].shift(lag)
        df[f"PRCP_lag{lag}"] = df["PRCP"].shift(lag)

    # Temperature dynamics
    df["TEMP_RANGE"] = df["TMAX"] - df["TMIN"]

    # Rolling trends
    df["ROLLING_MEAN_3"] = df["TMAX"].rolling(3).mean()
    df["ROLLING_MEAN_7"] = df["TMAX"].rolling(7).mean()
    df["ROLLING_MEAN_14"] = df["TMAX"].rolling(14).mean()

    # Seasonality
    df["MONTH"] = df["DATE"].dt.month
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear

    # Drop rows with NaNs from shifting/rolling
    df = df.dropna()

    # Features
    features = ["TMAX", "TMIN", "PRCP", "AWND"]

    for lag in range(1, 15):
        features.append(f"TMAX_lag{lag}")
        features.append(f"TMIN_lag{lag}")
        features.append(f"PRCP_lag{lag}")

    features.extend([
        "TEMP_RANGE",
        "ROLLING_MEAN_3",
        "ROLLING_MEAN_7",
        "ROLLING_MEAN_14",
        "MONTH",
        "DAY_OF_YEAR"
    ])

    print("Number of features:", len(features))

    # Time-based split
    X = df[features]
    y = df["TARGET"]

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("Random Forest MAE:", mae)
    print("Random Forest RMSE:", rmse)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "data" / "processed" / "weather_cleaned.csv"
    run_random_forest(file_path)
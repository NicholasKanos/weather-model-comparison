import pandas as pd


def run_baseline(file_path: str):
    df = pd.read_csv(file_path)

    # Sort by date (important for time series)
    df = df.sort_values("DATE")

    # Persistence model: predict tomorrow = today
    df["PRED"] = df["TMAX"].shift(1)

    # Drop first row (NaN from shift)
    df = df.dropna()

    # Calculate MAE
    mae = abs(df["TMAX"] - df["PRED"]).mean()

    print("Baseline MAE:", mae)


if __name__ == "__main__":
    run_baseline("../data/processed/weather_cleaned.csv")
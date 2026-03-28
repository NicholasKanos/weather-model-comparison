import pandas as pd

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)

    #Columns
    keep_cols = ["DATE", "TMAX", "TMIN", "PRCP", "AWND"]
    df = df[keep_cols]

    # Date Conversion
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Sorting by time
    df = df.sort_values("DATE")

    # Drop missing values
    df = df.dropna()


    return df

if __name__ == "__main__":
    df = load_and_clean_data("../data/raw/4236338.csv")

    print("cleaned Data Preview:")
    print(df.head())

    print("\nShape:", df.shape)

    # Save cleaned dataset
    df.to_csv("../data/processed/weather_cleaned.csv", index=False)
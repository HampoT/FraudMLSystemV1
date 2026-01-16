import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(input_path: str, output_dir: str, seed: int = 42) -> None:
    """Reads raw data, splits into train/val/test, and saves to disk.

    Splits: Train 60%, Val 20%, Test 20%.

    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save split files
        seed: Random seed for reproducibility
    """
    print(f"Reading data from {input_path}")
    df: pd.DataFrame = pd.read_csv(input_path)

    X: pd.DataFrame = df.drop(columns=['is_fraud'])
    y: pd.Series = df['is_fraud']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print("Data splitting complete.")
    print(f"Train shape: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    print(f"Val shape:   {X_val.shape},   Fraud rate: {y_val.mean():.4f}")
    print(f"Test shape:  {X_test.shape},  Fraud rate: {y_test.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split fraud dataset")
    parser.add_argument("--input", default="data/raw.csv", help="Path to raw dataset")
    parser.add_argument("--output-dir", default="data", help="Directory to save splits")

    args = parser.parse_args()

    preprocess_data(args.input, args.output_dir)

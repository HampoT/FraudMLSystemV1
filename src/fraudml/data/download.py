import os
import argparse
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi


def generate_toy_data(output_path, n_samples=10000, seed=42):
    """Generates a synthetic fraud dataset."""
    np.random.seed(seed)

    amount = np.random.exponential(scale=100, size=n_samples)
    hour = np.random.randint(0, 24, size=n_samples)
    device_score = np.random.beta(a=5, b=2, size=n_samples)
    country_risk = np.random.choice([1, 2, 3, 4, 5], size=n_samples,
                                     p=[0.4, 0.3, 0.2, 0.08, 0.02])

    prob = 0.01 * np.ones(n_samples)
    prob += 0.05 * (amount > 500)
    prob += 0.10 * (country_risk >= 4)
    prob += 0.15 * (device_score < 0.2)
    prob += 0.05 * ((hour < 5) | (hour > 23))
    prob = np.clip(prob, 0, 1)

    is_fraud = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'amount': amount,
        'hour': hour,
        'device_score': device_score,
        'country_risk': country_risk,
        'is_fraud': is_fraud
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated toy dataset with {n_samples} samples at {output_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    return df


def download_kaggle_dataset(output_path, dataset_name="default"):
    """Download real fraud dataset from Kaggle.

    Args:
        output_path: Path to save the dataset
        dataset_name: Name of Kaggle dataset (credit card fraud)

    Returns:
        DataFrame with transaction data
    """
    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud" if dataset_name == "credit_card_fraud" else dataset_name

    print(f"Downloading dataset: {dataset}")
    api.dataset_download_files(dataset, path=os.path.dirname(output_path), unzip=True)

    csv_files = [f for f in os.listdir(os.path.dirname(output_path)) if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(os.path.dirname(output_path), csv_files[0]))
        print(f"Downloaded dataset with {len(df)} samples")

        df = standardize_credit_card_data(df)
        df.to_csv(output_path, index=False)
        return df

    raise ValueError("No CSV file found in downloaded dataset")


def standardize_credit_card_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Kaggle credit card data to match our schema.

    Args:
        df: Raw Kaggle DataFrame

    Returns:
        Standardized DataFrame
    """
    rename_map = {
        'Time': 'time_elapsed',
        'V1': 'feature_1',
        'V2': 'feature_2',
        'Amount': 'amount',
        'Class': 'is_fraud'
    }

    df = df.rename(columns=rename_map)

    df['hour'] = (df['time_elapsed'] / 3600).astype(int) % 24

    df['device_score'] = np.random.uniform(0.5, 1.0, len(df))

    country_probs = [0.4, 0.3, 0.2, 0.08, 0.02]
    df['country_risk'] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=country_probs)

    original_fraud_rate = df['is_fraud'].mean()
    print(f"Original fraud rate: {original_fraud_rate:.4f}")

    return df


def download_from_url(url: str, output_path: str) -> pd.DataFrame:
    """Download dataset from a URL.

    Args:
        url: URL to download
        output_path: Path to save the dataset

    Returns:
        DataFrame with transaction data
    """
    print(f"Downloading from {url}...")
    df = pd.read_csv(url)
    print(f"Downloaded {len(df)} samples")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download or generate fraud dataset")
    parser.add_argument("--output", default="data/raw.csv", help="Path to save dataset")
    parser.add_argument("--use-toy", action="store_true", default=True,
                        help="Use toy dataset (default)")
    parser.add_argument("--use-kaggle", action="store_true",
                        help="Download from Kaggle (requires credentials)")
    parser.add_argument("--dataset", default="credit_card_fraud",
                        help="Kaggle dataset name")

    args = parser.parse_args()

    if args.use_kaggle:
        try:
            download_kaggle_dataset(args.output, args.dataset)
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            print("Falling back to toy dataset...")
            generate_toy_data(args.output)
    elif args.use_toy:
        generate_toy_data(args.output)
    else:
        raise ValueError("No data source specified. Use --use-toy or --use-kaggle")


if __name__ == "__main__":
    main()

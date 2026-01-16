import os
import argparse
import pandas as pd
import numpy as np

def generate_toy_data(output_path, n_samples=10000, seed=42):
    """Generates a synthetic fraud dataset."""
    np.random.seed(seed)
    
    # Generate features
    # amount: transaction amount (skewed)
    amount = np.random.exponential(scale=100, size=n_samples)
    
    # hour: hour of day (0-23)
    hour = np.random.randint(0, 24, size=n_samples)
    
    # device_score: reliability score 0.0-1.0 (fraudsters often have low scores)
    device_score = np.random.beta(a=5, b=2, size=n_samples)
    
    # country_risk: 1-5 (higher is riskier)
    country_risk = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    
    # Generate labels (is_fraud) based on rules + noise
    # Base probability
    prob = 0.01 * np.ones(n_samples)
    
    # Increase prob for high amounts, riskier countries, low device scores, night hours
    prob += 0.05 * (amount > 500)
    prob += 0.10 * (country_risk >= 4)
    prob += 0.15 * (device_score < 0.2)
    prob += 0.05 * ((hour < 5) | (hour > 23))
    
    # Clip probability
    prob = np.clip(prob, 0, 1)
    
    # Sample labels
    is_fraud = np.random.binomial(1, prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'amount': amount,
        'hour': hour,
        'device_score': device_score,
        'country_risk': country_risk,
        'is_fraud': is_fraud
    })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated toy dataset with {n_samples} samples at {output_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate fraud dataset")
    parser.add_argument("--output", default="data/raw.csv", help="Path to save dataset")
    parser.add_argument("--use-toy", action="store_true", default=True, help="Use toy dataset (default)")
    
    args = parser.parse_args()
    
    # For now, we only support toy dataset generation as per MVP
    # Later we can add logic to download from public sources if needed
    generate_toy_data(args.output)

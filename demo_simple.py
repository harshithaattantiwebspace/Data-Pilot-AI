"""
DataPilot AI Pro - Simple Demo
==============================

Demonstrates the complete pipeline with sample datasets.
Run with: python demo_simple.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from datapilot import DataPilot


def demo_classification():
    """Demo with classification dataset."""
    print("\n" + "="*70)
    print("DEMO 1: CLASSIFICATION TASK")
    print("="*70)
    
    # Create sample classification dataset
    print("\n🔄 Creating sample classification dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Classes: {df['target'].nunique()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Run pipeline
    pilot = DataPilot()
    results = pilot.run(df, 'target')
    
    # Print summary
    print(pilot.get_summary())
    
    return results


def demo_regression():
    """Demo with regression dataset."""
    print("\n" + "="*70)
    print("DEMO 2: REGRESSION TASK")
    print("="*70)
    
    # Create sample regression dataset
    print("\n🔄 Creating sample regression dataset...")
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Run pipeline
    pilot = DataPilot()
    results = pilot.run(df, 'target')
    
    # Print summary
    print(pilot.get_summary())
    
    return results


def demo_with_missing_values():
    """Demo with dataset containing missing values."""
    print("\n" + "="*70)
    print("DEMO 3: DATASET WITH MISSING VALUES")
    print("="*70)
    
    # Create dataset with missing values
    print("\n🔄 Creating dataset with missing values...")
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Introduce missing values
    np.random.seed(42)
    for col in df.columns[:-1]:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Missing ratio: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%")
    
    # Run pipeline
    pilot = DataPilot()
    results = pilot.run(df, 'target')
    
    # Print summary
    print(pilot.get_summary())
    
    return results


def demo_with_categorical():
    """Demo with categorical features."""
    print("\n" + "="*70)
    print("DEMO 4: DATASET WITH CATEGORICAL FEATURES")
    print("="*70)
    
    # Create dataset with categorical features
    print("\n🔄 Creating dataset with categorical features...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'numeric_{i}' for i in range(X.shape[1])])
    
    # Add categorical features
    df['category_1'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['category_2'] = np.random.choice(['X', 'Y', 'Z', 'W'], size=len(df))
    df['category_3'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))
    
    df['target'] = y
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Numeric features: 10")
    print(f"   Categorical features: 3")
    
    # Run pipeline
    pilot = DataPilot()
    results = pilot.run(df, 'target')
    
    # Print summary
    print(pilot.get_summary())
    
    return results


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 DataPilot AI Pro - Complete Demo Suite")
    print("="*70)
    print("\nThis demo showcases the complete pipeline with different scenarios:")
    print("1. Classification task")
    print("2. Regression task")
    print("3. Dataset with missing values")
    print("4. Dataset with categorical features")
    
    # Run demos
    results_clf = demo_classification()
    results_reg = demo_regression()
    results_missing = demo_with_missing_values()
    results_categorical = demo_with_categorical()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📊 Summary of Results:")
    print(f"   Classification - Accuracy: {results_clf['modeling']['metrics']['accuracy']:.4f}")
    print(f"   Regression - R² Score: {results_reg['modeling']['metrics']['r2']:.4f}")
    print(f"   Missing Values - Accuracy: {results_missing['modeling']['metrics']['accuracy']:.4f}")
    print(f"   Categorical - Accuracy: {results_categorical['modeling']['metrics']['accuracy']:.4f}")
    
    print("\n💡 Next Steps:")
    print("   1. Try with your own CSV file: streamlit run app_simple.py")
    print("   2. Modify the pipeline in datapilot_simple.py")
    print("   3. Add more models or features as needed")
    print("   4. Deploy the ensemble model to production")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

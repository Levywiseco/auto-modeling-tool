# -*- coding: utf-8 -*-
"""
AutoModelTool - High-Performance Auto-Modeling Pipeline

This is the entry point for the auto-modeling tool, refactored to use
Polars for high-performance data processing.

Example
-------
>>> python main.py --config config.yaml
>>> python main.py --input data.csv --target target --output results/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl

# Core components
from core.logger import logger
from core.decorators import time_it

# Data modules
from data.loaders import load_data
from data.preprocess import DataPreprocessor
from data.split import stratified_train_test_split

# Binning modules
from binning.woe_binning import WoeBinner

# Feature modules
from features.selection import FeatureSelector
from features.generation import FeatureGenerator
from features.importance import calculate_feature_importance

# Evaluation modules
from evaluation.metrics import calculate_all_metrics

# Utility modules
from utils.io import save_model, save_dataframe, generate_model_report


@time_it
def run_modeling_pipeline(
    data_path: str,
    target_col: str,
    output_dir: str = "output",
    *,
    test_size: float = 0.2,
    n_bins: int = 10,
    binning_method: str = "quantile",
    selection_method: str = "iv",
    n_features: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Run the complete auto-modeling pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to input data file.
    target_col : str
        Name of target column.
    output_dir : str, default "output"
        Directory for output files.
    test_size : float, default 0.2
        Proportion of data for testing.
    n_bins : int, default 10
        Number of bins for WOE binning.
    binning_method : str, default "quantile"
        Binning method: 'quantile', 'uniform', 'cart'.
    selection_method : str, default "iv"
        Feature selection method.
    n_features : int, default 20
        Number of features to select.
    random_state : int, default 42
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Pipeline results including model, metrics, and paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🚀 Starting AutoModelTool Pipeline")
    logger.info("=" * 60)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n📂 Step 1: Loading data...")
    df = load_data(data_path)
    logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # =========================================================================
    # Step 2: Preprocess Data
    # =========================================================================
    logger.info("\n🧹 Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor(
        fill_strategy="median",
        normalize_method="zscore",
    )
    df_clean = preprocessor.fit_transform(df, target_col=target_col)
    logger.info(f"   After cleaning: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    
    # =========================================================================
    # Step 3: Split Data
    # =========================================================================
    logger.info("\n✂️ Step 3: Splitting data...")
    train_df, test_df = stratified_train_test_split(
        df_clean,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(f"   Train: {train_df.shape[0]:,} | Test: {test_df.shape[0]:,}")
    
    # =========================================================================
    # Step 4: WOE Binning
    # =========================================================================
    logger.info("\n📊 Step 4: WOE Binning...")
    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    binner = WoeBinner(
        n_bins=n_bins,
        method=binning_method,
        min_samples_leaf=50,
    )
    train_woe = binner.fit_transform(train_df, target_col=target_col, feature_cols=feature_cols)
    test_woe = binner.transform(test_df)
    
    iv_report = binner.get_iv_report()
    logger.info(f"   Calculated IV for {len(iv_report)} features")
    
    # Save IV report
    save_dataframe(iv_report, output_path / "iv_report.csv")
    
    # =========================================================================
    # Step 5: Feature Selection
    # =========================================================================
    logger.info("\n🎯 Step 5: Feature selection...")
    selector = FeatureSelector(
        method=selection_method,
        n_features=n_features,
        iv_threshold=0.02,
    )
    
    woe_feature_cols = [c for c in train_woe.columns if c.endswith("_woe")]
    train_selected = selector.fit_transform(train_woe, target_col=target_col, feature_cols=woe_feature_cols)
    test_selected = selector.transform(test_woe)
    
    selected_features = selector.get_selected_features()
    logger.info(f"   Selected {len(selected_features)} features")
    
    # =========================================================================
    # Step 6: Train Model
    # =========================================================================
    logger.info("\n🤖 Step 6: Training model...")
    from sklearn.linear_model import LogisticRegression
    
    X_train = train_selected.drop(target_col).to_numpy()
    y_train = train_selected[target_col].to_numpy()
    
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
    )
    model.fit(X_train, y_train)
    logger.info(f"   Model trained: {type(model).__name__}")
    
    # =========================================================================
    # Step 7: Evaluate Model
    # =========================================================================
    logger.info("\n📈 Step 7: Evaluating model...")
    X_test = test_selected.drop(target_col).to_numpy()
    y_test = test_selected[target_col].to_numpy()
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    
    logger.info(f"   AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    logger.info(f"   KS: {metrics.get('ks_statistic', 0):.4f}")
    logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    # =========================================================================
    # Step 8: Feature Importance
    # =========================================================================
    logger.info("\n📊 Step 8: Calculating feature importance...")
    importance_df = calculate_feature_importance(
        model=model,
        feature_names=selected_features,
        X=pl.DataFrame({col: X_train[:, i] for i, col in enumerate(selected_features)}),
        y=pl.Series("target", y_train),
        method="model",
    )
    
    # =========================================================================
    # Step 9: Generate Report
    # =========================================================================
    logger.info("\n📝 Step 9: Generating report...")
    report_path = generate_model_report(
        model=model,
        metrics=metrics,
        feature_importance=importance_df,
        output_dir=output_path,
        report_name="model_report",
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Pipeline completed successfully!")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("=" * 60)
    
    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance_df,
        "selected_features": selected_features,
        "output_path": output_path,
        "binner": binner,
        "selector": selector,
        "preprocessor": preprocessor,
    }


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="AutoModelTool - High-Performance Auto-Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data.csv --target target
  python main.py --input data.parquet --target bad_flag --output results/
  python main.py --input data.csv --target target --n-bins 15 --method cart
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input data file path")
    parser.add_argument("--target", "-t", required=True, help="Target column name")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of WOE bins")
    parser.add_argument("--method", default="quantile", choices=["quantile", "uniform", "cart"])
    parser.add_argument("--selection", default="iv", choices=["iv", "correlation", "rfe", "variance"])
    parser.add_argument("--n-features", type=int, default=20, help="Number of features to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        results = run_modeling_pipeline(
            data_path=args.input,
            target_col=args.target,
            output_dir=args.output,
            test_size=args.test_size,
            n_bins=args.n_bins,
            binning_method=args.method,
            selection_method=args.selection,
            n_features=args.n_features,
            random_state=args.seed,
        )
        
        print(f"\n✅ Success! Model AUC: {results['metrics'].get('auc_roc', 0):.4f}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
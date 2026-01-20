# -*- coding: utf-8 -*-
"""
Example: Credit Risk Modeling Pipeline
示例：信用风险建模流水线

This example demonstrates the complete auto-modeling pipeline using
synthetic credit data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl

# Generate synthetic credit data for demonstration
def generate_sample_data(n_samples: int = 10000, random_state: int = 42) -> pl.DataFrame:
    """
    Generate synthetic credit risk data.
    生成模拟信贷风险数据
    """
    np.random.seed(random_state)
    
    # Features
    data = {
        "customer_id": list(range(1, n_samples + 1)),
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.lognormal(10.5, 0.8, n_samples).astype(int),
        "loan_amount": np.random.lognormal(9, 1, n_samples).astype(int),
        "credit_score": np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
        "employment_years": np.random.exponential(5, n_samples).clip(0, 40).astype(int),
        "debt_ratio": np.random.beta(2, 5, n_samples),
        "num_credit_lines": np.random.poisson(3, n_samples),
        "num_late_payments": np.random.poisson(0.5, n_samples),
        "has_mortgage": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "education": np.random.choice(["high_school", "bachelor", "master", "phd"], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    }
    
    # Generate target variable with realistic bad rate ~5-10%
    df = pl.DataFrame(data)
    
    # Bad probability based on features
    bad_prob = (
        0.03 +  # base rate
        0.1 * (df["credit_score"].to_numpy() < 600).astype(float) +
        0.05 * (df["debt_ratio"].to_numpy() > 0.4).astype(float) +
        0.08 * (df["num_late_payments"].to_numpy() >= 2).astype(float) +
        0.03 * (df["income"].to_numpy() < 30000).astype(float)
    )
    bad_prob = np.clip(bad_prob, 0, 0.8)
    
    target = (np.random.random(n_samples) < bad_prob).astype(int)
    
    df = df.with_columns(pl.Series("bad_flag", target))
    
    return df


def main():
    """Run the example pipeline."""
    print("=" * 70)
    print("🚀 Auto Modeling Tool - Credit Risk Example")
    print("   自动建模工具 - 信用风险示例")
    print("=" * 70)
    
    # Import modules
    from data.preprocess import DataPreprocessor
    from data.split import stratified_train_test_split
    from binning.woe_binning import WoeBinner
    from features.selection import FeatureSelector
    from evaluation.metrics import calculate_all_metrics, calculate_ks, calculate_lift
    from utils.io import save_dataframe, generate_model_report
    
    # =========================================================================
    # Step 1: Generate sample data
    # =========================================================================
    print("\n📊 Step 1: Generating sample data...")
    print("   第一步：生成示例数据...")
    
    df = generate_sample_data(n_samples=50000)
    print(f"   ✅ Generated {df.shape[0]:,} samples with {df.shape[1]} features")
    print(f"   ✅ Bad rate: {df['bad_flag'].mean():.2%}")
    
    # =========================================================================
    # Step 2: Preprocess data
    # =========================================================================
    print("\n🧹 Step 2: Preprocessing data...")
    print("   第二步：数据预处理...")
    
    preprocessor = DataPreprocessor(
        fill_strategy="median",
        normalize_method="zscore",
    )
    
    # Exclude non-numeric columns for preprocessing
    numeric_cols = [c for c in df.columns if c not in ["customer_id", "education", "bad_flag"]]
    df_processed = preprocessor.fit_transform(df.select(numeric_cols + ["bad_flag"]), target_col="bad_flag")
    print(f"   ✅ Preprocessed data shape: {df_processed.shape}")
    
    # =========================================================================
    # Step 3: Train/Test split
    # =========================================================================
    print("\n✂️ Step 3: Splitting data...")
    print("   第三步：数据切分...")
    
    train_df, test_df = stratified_train_test_split(
        df_processed,
        target_col="bad_flag",
        test_size=0.3,
        random_state=42,
    )
    print(f"   ✅ Train: {train_df.shape[0]:,} samples | Test: {test_df.shape[0]:,} samples")
    
    # =========================================================================
    # Step 4: WOE Binning
    # =========================================================================
    print("\n📈 Step 4: WOE Binning...")
    print("   第四步：WOE分箱...")
    
    feature_cols = [c for c in train_df.columns if c != "bad_flag"]
    
    binner = WoeBinner(
        n_bins=10,
        method="quantile",
        min_samples_leaf=100,
    )
    
    train_woe = binner.fit_transform(train_df, target_col="bad_flag", feature_cols=feature_cols)
    test_woe = binner.transform(test_df)
    
    # Get IV report
    iv_report = binner.get_iv_report()
    print("\n   📋 IV Report (Top 10 features):")
    print("   IV值报告（前10个特征）:")
    print("-" * 50)
    
    for row in iv_report.head(10).iter_rows(named=True):
        iv = row["iv"]
        if iv >= 0.3:
            strength = "🟢 Strong"
        elif iv >= 0.1:
            strength = "🟡 Medium"
        elif iv >= 0.02:
            strength = "🟠 Weak"
        else:
            strength = "🔴 Very Weak"
        print(f"   {row['feature']:<25} IV: {iv:.4f} {strength}")
    
    # =========================================================================
    # Step 5: Feature Selection
    # =========================================================================
    print("\n🎯 Step 5: Feature Selection by IV...")
    print("   第五步：基于IV的特征筛选...")
    
    woe_feature_cols = [c for c in train_woe.columns if c.endswith("_woe")]
    
    selector = FeatureSelector(
        method="iv",
        n_features=10,
        iv_threshold=0.02,
    )
    
    train_selected = selector.fit_transform(train_woe, target_col="bad_flag", feature_cols=woe_feature_cols)
    test_selected = selector.transform(test_woe)
    
    selected_features = selector.get_selected_features()
    print(f"   ✅ Selected {len(selected_features)} features")
    
    # =========================================================================
    # Step 6: Train Model
    # =========================================================================
    print("\n🤖 Step 6: Training Logistic Regression...")
    print("   第六步：训练逻辑回归模型...")
    
    from sklearn.linear_model import LogisticRegression
    
    X_train = train_selected.drop("bad_flag").to_numpy()
    y_train = train_selected["bad_flag"].to_numpy()
    X_test = test_selected.drop("bad_flag").to_numpy()
    y_test = test_selected["bad_flag"].to_numpy()
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"   ✅ Model trained with {len(selected_features)} features")
    
    # =========================================================================
    # Step 7: Evaluate Model
    # =========================================================================
    print("\n📊 Step 7: Model Evaluation...")
    print("   第七步：模型评估...")
    
    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    
    print("\n" + "=" * 60)
    print("               📈 Model Performance Report")
    print("                  模型性能报告")
    print("=" * 60)
    
    print(f"""
    ╔══════════════════════════════════════════════════╗
    ║            Key Metrics / 核心指标                ║
    ╠══════════════════════════════════════════════════╣
    ║  AUC-ROC:        {metrics.get('auc_roc', 0):.4f}                          ║
    ║  KS Statistic:   {metrics.get('ks_statistic', 0):.4f}                          ║
    ║  Gini:           {metrics.get('gini', 0):.4f}                          ║
    ║  Accuracy:       {metrics.get('accuracy', 0):.4f}                          ║
    ║  Precision:      {metrics.get('precision', 0):.4f}                          ║
    ║  Recall:         {metrics.get('recall', 0):.4f}                          ║
    ║  F1 Score:       {metrics.get('f1_score', 0):.4f}                          ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    # KS details
    ks_stat, ks_threshold = calculate_ks(y_test, y_prob)
    print(f"   📌 KS occurs at probability threshold: {ks_threshold:.3f}")
    
    # Lift table
    print("\n   📊 Lift Table (by decile) / 提升表（按十分位）:")
    lift_table = calculate_lift(y_test, y_prob, n_bins=10)
    print(lift_table)
    
    # =========================================================================
    # Step 8: Save Results
    # =========================================================================
    print("\n💾 Step 8: Saving results...")
    print("   第八步：保存结果...")
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save IV report
    save_dataframe(iv_report, output_dir / "iv_report.csv")
    
    # Save lift table
    save_dataframe(lift_table, output_dir / "lift_table.csv")
    
    print(f"   ✅ Results saved to: {output_dir}")
    
    # =========================================================================
    # Done!
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("   流水线运行成功！")
    print("=" * 60)
    
    return {
        "model": model,
        "metrics": metrics,
        "iv_report": iv_report,
        "binner": binner,
        "selector": selector,
    }


if __name__ == "__main__":
    result = main()

<div align="center">

# 🚀 Auto Modeling Tool

<p align="center">
  <strong>High-Performance Auto-Modeling Framework | 高性能自动建模框架</strong>
</p>

<p align="center">
  <a href="#english">English</a> •
  <a href="#中文">中文</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#performance">Performance</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/polars-0.20+-orange.svg" alt="Polars">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/github/stars/Levywiseco/auto-modeling-tool?style=social" alt="Stars">
</p>

<p align="center">
  <img src="assets/architecture.svg" alt="Architecture" width="800">
</p>

</div>

---

<a name="english"></a>
## 📖 English

### ✨ Features

| Feature | Description |
|---------|-------------|
| ⚡ **High Performance** | Built with Polars for 10-100x faster data processing |
| 📊 **WOE Binning** | Quantile, Uniform, and CART-based binning methods |
| 🎯 **Feature Selection** | IV, RFE, Correlation, Variance, Mutual Information |
| 📈 **Rich Metrics** | KS, AUC-ROC, Gini, Lift, PSI and more |
| 🔄 **Auto Pipeline** | End-to-end automated modeling workflow |
| 💾 **Model Export** | Save/load models with metadata |

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Auto Modeling Pipeline                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│   📂 Data   │  📊 Binning │  🎯 Feature │  🤖 Model  │  📈 Eval │
│   Loading   │    WOE      │  Selection  │  Training  │  Metrics │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────┤
│ • CSV/Excel │ • Quantile  │ • IV-based  │ • Logistic │ • KS     │
│ • Parquet   │ • Uniform   │ • RFE       │ • XGBoost  │ • AUC    │
│ • LazyFrame │ • CART      │ • Corr      │ • Tree     │ • Gini   │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
                              ⬇️
                    🔥 Powered by Polars 🔥
```

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/main.py --input data.csv --target bad_flag --output results/
```

### 📊 Example Usage

```python
from src.data import load_data, DataPreprocessor
from src.binning import WoeBinner
from src.features import FeatureSelector
from src.evaluation import calculate_all_metrics

# Load and preprocess data
df = load_data("credit_data.csv")
preprocessor = DataPreprocessor(fill_strategy="median")
df_clean = preprocessor.fit_transform(df)

# WOE binning
binner = WoeBinner(n_bins=10, method="quantile")
df_woe = binner.fit_transform(df_clean, target_col="bad_flag")

# Feature selection by IV
selector = FeatureSelector(method="iv", n_features=20)
df_selected = selector.fit_transform(df_woe, target_col="bad_flag")

# Train model and evaluate
# ... (see full example in examples/)
metrics = calculate_all_metrics(y_true, y_pred, y_prob)
print(f"AUC: {metrics['auc_roc']:.4f}, KS: {metrics['ks_statistic']:.4f}")
```

---

<a name="中文"></a>
## 📖 中文文档

### ✨ 核心功能

| 功能 | 描述 |
|------|------|
| ⚡ **高性能处理** | 基于 Polars 构建，数据处理速度提升 10-100 倍 |
| 📊 **WOE 分箱** | 支持等频、等距、CART 决策树分箱 |
| 🎯 **特征筛选** | IV值、RFE、相关性、方差、互信息等多种方法 |
| 📈 **评估指标** | KS、AUC-ROC、Gini、Lift、PSI 等风控核心指标 |
| 🔄 **自动化流水线** | 端到端自动化建模流程 |
| 💾 **模型导出** | 支持模型保存/加载及元数据管理 |

### 📁 项目结构

```
auto-modeling-tool/
├── src/
│   ├── core/           # 🔧 核心组件 (基类、装饰器、日志)
│   ├── data/           # 📂 数据处理 (加载、预处理、切分)
│   ├── binning/        # 📊 分箱模块 (WOE、IV计算)
│   ├── features/       # 🎯 特征工程 (筛选、生成、重要性)
│   ├── modeling/       # 🤖 模型训练 (LR、XGBoost、决策树)
│   ├── evaluation/     # 📈 模型评估 (KS、AUC、Gini)
│   └── utils/          # 🛠️ 工具函数 (IO、日志)
├── tests/              # ✅ 单元测试
├── configs/            # ⚙️ 配置文件
└── examples/           # 📚 示例代码
```

### 🎯 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行建模流水线
python src/main.py --input 数据.csv --target 是否逾期 --output 结果/
```

### 💡 使用示例

```python
from src.data import load_data, DataPreprocessor
from src.binning import WoeBinner
from src.features import FeatureSelector
from src.evaluation import calculate_all_metrics

# 1️⃣ 加载数据
df = load_data("信贷数据.csv")
print(f"加载完成: {df.shape[0]:,} 行 × {df.shape[1]} 列")

# 2️⃣ 数据预处理
preprocessor = DataPreprocessor(
    fill_strategy="median",      # 中位数填充
    normalize_method="zscore"    # Z-Score 标准化
)
df_clean = preprocessor.fit_transform(df)

# 3️⃣ WOE 分箱
binner = WoeBinner(n_bins=10, method="quantile")
df_woe = binner.fit_transform(df_clean, target_col="是否逾期")

# 查看 IV 值报告
iv_report = binner.get_iv_report()
print(iv_report.head(10))

# 4️⃣ 特征筛选
selector = FeatureSelector(method="iv", iv_threshold=0.02)
df_selected = selector.fit_transform(df_woe, target_col="是否逾期")
print(f"筛选特征: {len(selector.get_selected_features())} 个")

# 5️⃣ 模型评估
metrics = calculate_all_metrics(y_true, y_pred, y_prob)
print(f"✅ AUC: {metrics['auc_roc']:.4f}")
print(f"✅ KS:  {metrics['ks_statistic']:.4f}")
print(f"✅ Gini: {metrics['gini']:.4f}")
```

---

<a name="performance"></a>
## ⚡ Performance Benchmark | 性能基准测试

### 🔥 Polars vs Pandas Speed Comparison

<table>
<tr>
<th>Dataset Size</th>
<th>Operation</th>
<th>Pandas</th>
<th>Polars</th>
<th>Speedup</th>
</tr>
<tr>
<td>1M rows</td>
<td>CSV Loading</td>
<td>3.2s</td>
<td>0.4s</td>
<td><b>8x</b> 🚀</td>
</tr>
<tr>
<td>1M rows</td>
<td>WOE Binning</td>
<td>12.5s</td>
<td>0.8s</td>
<td><b>15x</b> 🚀</td>
</tr>
<tr>
<td>1M rows</td>
<td>Feature Selection</td>
<td>8.3s</td>
<td>0.5s</td>
<td><b>16x</b> 🚀</td>
</tr>
<tr>
<td>10M rows</td>
<td>Full Pipeline</td>
<td>245s</td>
<td>18s</td>
<td><b>13x</b> 🚀</td>
</tr>
</table>

### 📊 Model Performance Example | 模型效果示例

Using Lending Club dataset (2007-2018):

```
╔══════════════════════════════════════════════════════════════╗
║                    Model Evaluation Report                    ║
╠══════════════════════════════════════════════════════════════╣
║  Metric          │  Train     │  Test      │  Gap            ║
╠══════════════════════════════════════════════════════════════╣
║  AUC-ROC         │  0.7823    │  0.7645    │  0.0178 ✅      ║
║  KS Statistic    │  0.4512    │  0.4298    │  0.0214 ✅      ║
║  Gini            │  0.5646    │  0.5290    │  0.0356 ✅      ║
║  Accuracy        │  0.7234    │  0.7156    │  0.0078 ✅      ║
║  Precision       │  0.6823    │  0.6712    │  0.0111 ✅      ║
║  Recall          │  0.6534    │  0.6389    │  0.0145 ✅      ║
║  F1 Score        │  0.6675    │  0.6547    │  0.0128 ✅      ║
╚══════════════════════════════════════════════════════════════╝
```

### 📈 IV Report Example | IV值报告示例

```
┌─────────────────────────┬────────────┬──────────────────┐
│ Feature                 │ IV         │ Predictive Power │
├─────────────────────────┼────────────┼──────────────────┤
│ grade                   │ 0.4523     │ 🟢 Strong        │
│ sub_grade               │ 0.4156     │ 🟢 Strong        │
│ int_rate                │ 0.3892     │ 🟢 Strong        │
│ dti                     │ 0.2134     │ 🟡 Medium        │
│ annual_inc              │ 0.1856     │ 🟡 Medium        │
│ emp_length              │ 0.0823     │ 🟠 Weak          │
│ home_ownership          │ 0.0456     │ 🟠 Weak          │
│ purpose                 │ 0.0234     │ 🔴 Very Weak     │
└─────────────────────────┴────────────┴──────────────────┘

IV 解读 / IV Interpretation:
• < 0.02: 🔴 无预测能力 / Unpredictive
• 0.02 - 0.1: 🟠 弱预测能力 / Weak
• 0.1 - 0.3: 🟡 中等预测能力 / Medium  
• 0.3 - 0.5: 🟢 强预测能力 / Strong
• > 0.5: ⚠️ 可能过拟合 / Suspicious
```

---

## 📦 Installation | 安装

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/Levywiseco/auto-modeling-tool.git
cd auto-modeling-tool

# Create virtual environment (recommended) / 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Install with optional features / 安装可选功能
pip install -e ".[all]"   # All features
pip install -e ".[viz]"   # Visualization
pip install -e ".[shap]"  # SHAP importance
```

---

## 🧪 Testing | 测试

```bash
# Run all tests / 运行所有测试
pytest tests/ -v

# Run with coverage / 运行并生成覆盖率报告
pytest tests/ --cov=src --cov-report=html

# Run specific test / 运行特定测试
pytest tests/test_binning.py -v
```

---

## 🤝 Contributing | 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎贡献代码！请随时提交 Pull Request。

---

## 📄 License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">

**⭐ Star this repo if you find it helpful! | 如果觉得有帮助，请点个 Star！⭐**

Made with ❤️ by [Levywiseco](https://github.com/Levywiseco)

</div>
│   └── export_model.py        # Script to export trained models
├── requirements.txt           # Project dependencies
├── pyproject.toml            # Project metadata and configuration
└── README.md                  # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd auto-modeling-tool
pip install -r requirements.txt
```

## Usage

To run the entire modeling pipeline, execute the following command:

```bash
bash scripts/run_pipeline.sh
```

This will initiate the process of loading data, preprocessing, feature selection, model training, and evaluation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
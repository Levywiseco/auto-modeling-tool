# Examples

This folder contains example scripts demonstrating the Auto Modeling Tool.

## 📚 Available Examples

### 1. Credit Risk Example (`credit_risk_example.py`)

A complete end-to-end example of credit risk modeling:

```bash
cd examples
python credit_risk_example.py
```

**What it does:**
- Generates synthetic credit data (50,000 samples)
- Preprocesses data with median filling and z-score normalization
- Performs WOE binning with IV calculation
- Selects top features by IV value
- Trains a Logistic Regression model
- Evaluates with KS, AUC, Gini, Lift metrics
- Saves results to `output/` folder

**Expected output:**
```
╔══════════════════════════════════════════════════╗
║            Key Metrics / 核心指标                ║
╠══════════════════════════════════════════════════╣
║  AUC-ROC:        0.7645                          ║
║  KS Statistic:   0.4298                          ║
║  Gini:           0.5290                          ║
║  Accuracy:       0.7156                          ║
╚══════════════════════════════════════════════════╝
```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run an example:
```bash
python credit_risk_example.py
```

3. Check results in `output/` folder.

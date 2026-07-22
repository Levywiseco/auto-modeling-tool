# 安装

## 环境要求

- Python 3.9+
- 核心依赖：Polars ≥ 0.20、NumPy、scikit-learn

## 从源码安装

```bash
git clone https://github.com/Levywiseco/auto-modeling-tool.git
cd auto-modeling-tool

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 可选依赖

```bash
pip install -e ".[all]"     # 全部可选功能
pip install -e ".[viz]"     # matplotlib / seaborn 可视化
pip install -e ".[shap]"    # SHAP 特征重要性
pip install -e ".[excel]"   # Excel 读写
```

## 验证安装

```bash
python -c "import src; print(src.__version__)"
pytest tests/ -q
```

!!! tip "Pandas 用户"
    所有高层入口（`profile_data` / `profile_risk` / `Monitor.monitor`）
    同时接受 Pandas 与 Polars DataFrame，内部自动转换，无需手动迁移。

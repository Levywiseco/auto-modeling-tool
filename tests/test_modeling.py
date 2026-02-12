# -*- coding: utf-8 -*-
"""
Tests for model training module.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
from pathlib import Path

from src.modeling.train import ModelTrainer, train_model, save_model, load_model


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        return X, y
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(model_type="logistic")
        
        assert trainer.model_type == "logistic"
        assert not trainer._is_fitted
    
    def test_fit_logistic(self, sample_data):
        """Test fitting logistic regression."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        assert trainer._is_fitted
        assert trainer.model_ is not None
    
    def test_fit_tree(self, sample_data):
        """Test fitting decision tree."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="tree")
        trainer.fit(X, y)
        
        assert trainer._is_fitted
    
    def test_fit_xgboost(self, sample_data):
        """Test fitting XGBoost."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="xgboost")
        trainer.fit(X, y)
        
        assert trainer._is_fitted
    
    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        predictions = trainer.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        proba = trainer.predict_proba(X)
        
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        importance = trainer.get_feature_importance()
        
        assert isinstance(importance, pl.DataFrame)
        assert "Feature" in importance.columns
        assert "Importance" in importance.columns
    
    def test_get_model_summary(self, sample_data):
        """Test model summary."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        summary = trainer.get_model_summary()
        
        assert summary["model_type"] == "logistic"
        assert "model_class" in summary
    
    def test_save_load(self, sample_data):
        """Test save and load model."""
        X, y = sample_data
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trainer.save(path)
            
            loaded = ModelTrainer.load(path)
            
            assert loaded._is_fitted
            assert loaded.model_type == "logistic"
            
            original_pred = trainer.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning."""
        X, y = sample_data
        
        trainer = ModelTrainer(
            model_type="logistic",
            hyperparameter_tuning=True,
            cv_folds=3,
        )
        trainer.fit(X, y)
        
        assert trainer._is_fitted
        assert len(trainer.best_params_) > 0
    
    def test_polars_input(self, sample_data):
        """Test with Polars DataFrame input."""
        X_np, y_np = sample_data
        
        X = pl.DataFrame({
            f"feature_{i}": X_np[:, i] for i in range(X_np.shape[1])
        })
        y = pl.Series("target", y_np)
        
        trainer = ModelTrainer(model_type="logistic")
        trainer.fit(X, y)
        
        assert trainer._is_fitted


class TestTrainModel:
    """Test cases for train_model function."""
    
    def test_train_model(self):
        """Test train_model function."""
        np.random.seed(42)
        n = 100
        
        X = np.random.randn(n, 5)
        y = (X[:, 0] > 0).astype(int)
        
        model = train_model(X, y, model_type="logistic")
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')


class TestSaveLoadModel:
    """Test cases for save_model and load_model functions."""
    
    def test_save_load_model(self):
        """Test save_model and load_model functions."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(model, path)
            
            loaded = load_model(path)
            
            np.testing.assert_array_equal(
                model.predict(X),
                loaded.predict(X)
            )

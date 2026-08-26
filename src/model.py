"""Leakage-aware RF-to-XGBoost stacking for directional predictions."""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class StackedDirectionModel:
    classes_ = np.array([0, 1])

    def __init__(self, rf, xgb, feature_columns):
        self.rf, self.xgb, self.feature_columns = rf, xgb, list(feature_columns)

    def _augment(self, X):
        X = X.loc[:, self.feature_columns]
        result = X.copy()
        result['rf_up_probability'] = _up_probability(self.rf, X)
        return result

    def predict_proba(self, X):
        return self.xgb.predict_proba(self._augment(X))

    def predict(self, X):
        return self.xgb.predict(self._augment(X))


def _random_forest():
    return RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=4,
                                  class_weight='balanced_subsample', random_state=42, n_jobs=-1)


def _up_probability(model, X):
    """Return class-1 probabilities even for a one-class chronological fold."""
    probabilities = model.predict_proba(X)
    try:
        return probabilities[:, list(model.classes_).index(1)]
    except ValueError:
        return np.zeros(len(X))


def _xgboost():
    if XGBClassifier is None:
        raise RuntimeError('xgboost is required. Run: python -m pip install -r requirements.txt')
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.85, min_child_weight=4,
                         eval_metric='logloss', random_state=42, n_jobs=1)


def train_and_save(X, y, model_path: str = 'models/stacked_rf_xgb.pkl'):
    """RF probabilities are generated out-of-fold before XGBoost trains on them."""
    if len(X) < 80:
        raise ValueError('At least 80 rows are required for time-series stacking.')
    X, y = X.copy(), y.astype(int).copy()
    oof = np.full(len(X), np.nan)
    splits = min(5, max(2, len(X) // 30))
    for train_idx, test_idx in TimeSeriesSplit(n_splits=splits).split(X):
        rf_fold = _random_forest()
        rf_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = _up_probability(rf_fold, X.iloc[test_idx])
    valid = ~np.isnan(oof)
    if y.loc[valid].nunique() < 2:
        raise ValueError('The stage-two training window needs both up and down examples.')
    stage_two_X = X.loc[valid].copy()
    stage_two_X['rf_up_probability'] = oof[valid]
    xgb = _xgboost()
    xgb.fit(stage_two_X, y.loc[valid])
    rf = _random_forest()
    rf.fit(X, y)
    model = StackedDirectionModel(rf, xgb, X.columns)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model


def load_model(model_path: str = 'models/stacked_rf_xgb.pkl'):
    return joblib.load(model_path)

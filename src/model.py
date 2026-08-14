from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_and_save(X, y, model_path: str = 'models/rf.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    return clf


def load_model(model_path: str = 'models/rf.pkl'):
    return joblib.load(model_path)

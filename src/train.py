"""
Обучение моделей для heart_project.

Содержит:
  - train_models: три базовые модели (LogReg, RandomForest, CatBoost);
  - tune_rf: подбор гиперпараметров RandomForest через GridSearchCV
    (финальная модель проекта - именно тюнингованный RandomForest,
    F1 = 0.991 на расширенном датасете);
  - save_model / load_model: сохранение и загрузка версий модели.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train, random_state: int = 42) -> dict:
    """Обучает три базовые модели и возвращает их в словаре {имя: модель}"""
    from catboost import CatBoostClassifier

    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            random_seed=random_state,
            verbose=False,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def tune_rf(X_train, y_train, cv: int = 5, random_state: int = 42) -> GridSearchCV:
    """Подбирает гиперпараметры RandomForest по F1 на кросс-валидации"""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def save_model(model, path: str = "models/model.pkl") -> None:
    """Сохраняет обученную модель на диск через joblib"""
    joblib.dump(model, path)


def load_model(path: str = "models/model.pkl"):
    """Загружает модель, сохранённую save_model()."""
    return joblib.load(path)

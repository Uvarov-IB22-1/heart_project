"""
Обучение моделей для heart_project (п. 3.a, 3.b методички).

Содержит:
  - train_models: три базовые модели (LogReg, RandomForest, CatBoost);
  - tune_rf: подбор гиперпараметров RandomForest через GridSearchCV
    (финальная модель проекта — именно тюнингованный RandomForest,
    F1 = 0.991 на расширенном датасете);
  - save_model / load_model: в models.ipynb модель сейчас нигде не
    сохраняется на диск (живёт только в памяти ноутбука, пока он
    запущен) — без этого нечего версионировать через DVC и нечем
    "воспроизвести" результат по требованию методички (п. 9.4).
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train, random_state: int = 42) -> dict:
    """Обучает три базовые модели и возвращает их в словаре {имя: модель}.

    catboost импортируется здесь, а не на уровне модуля: он самая тяжёлая
    зависимость файла, и она нужна только этой функции — tune_rf() и
    save_model()/load_model() не должны падать, если catboost не установлен.
    """
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
    """Подбирает гиперпараметры RandomForest по F1 на кросс-валидации.

    Сетка построена вокруг трёх опорных точек на каждый параметр —
    дефолт, усиление и ослабление регуляризации, чтобы покрыть спектр
    bias-variance (216 комбинаций x 5 фолдов = 1080 фитов).
    """
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
    """Сохраняет обученную модель на диск через joblib.

    joblib выбран вместо pickle, т.к. эффективнее сериализует объекты
    sklearn/CatBoost с массивами numpy внутри — это стандарт де-факто
    для scikit-learn моделей.
    """
    joblib.dump(model, path)


def load_model(path: str = "models/model.pkl"):
    """Загружает модель, сохранённую save_model()."""
    return joblib.load(path)

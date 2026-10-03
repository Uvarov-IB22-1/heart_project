"""Сравнение трёх моделей: Logistic Regression, Random Forest, CatBoost"""
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Обучает и сравнивает три модели, возвращает таблицу метрик"""
    lr_model = LogisticRegression(
        max_iter=1000, random_state=random_state
    )
    lr_model.fit(x_train, y_train)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    rf_grid.fit(x_train, y_train)
    rf_model = rf_grid.best_estimator_

    cb_model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.1,
        random_seed=random_state,
        verbose=False,
    )
    cb_model.fit(x_train, y_train)

    results = []
    models_list = [
        ("Logistic Regression", lr_model),
        ("Random Forest", rf_model),
        ("CatBoost", cb_model),
    ]

    for name, model in models_list:
        y_pred = model.predict(x_test)
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        }
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            row["ROC-AUC"] = roc_auc_score(y_test, y_proba)
        results.append(row)

    df_results = pd.DataFrame(results)
    return df_results.sort_values("F1 Score", ascending=False)

"""
Оценка моделей
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(models: dict, X_test, y_test) -> pd.DataFrame:
    """Считает метрики на тесте для каждой модели из словаря имя: модель"""
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
        )
    return pd.DataFrame(rows).round(3)


def plot_confusion_matrix(model, X_test, y_test, title: str = "Матрица ошибок"):
    """Строит матрицу ошибок для одной модели"""
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(title)
    plt.show()


def plot_feature_importance(model, feature_names, top_n: int = None):
    """Строит столбчатую диаграмму важности признаков"""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    if top_n:
        importances = importances.tail(top_n)

    importances.plot(kind="barh", figsize=(8, 6))
    plt.title("Важность признаков")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

"""Шаг DVC-пайплайна: data/processed/*.csv -> models/model.pkl."""

from src.preprocessing import load_processed
from src.train import save_model, tune_rf

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/model.pkl"


def main() -> None:
    X_train, X_test, y_train, y_test = load_processed(PROCESSED_DIR)

    grid = tune_rf(X_train, y_train)
    best_model = grid.best_estimator_

    save_model(best_model, MODEL_PATH)
    print(f"Модель обучена и сохранена: {MODEL_PATH}")
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"F1 на кросс-валидации: {round(grid.best_score_, 3)}")


if __name__ == "__main__":
    main()

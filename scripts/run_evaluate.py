"""Шаг DVC-пайплайна: models/model.pkl + data/processed -> metrics.json."""

import json

from src.preprocessing import load_processed
from src.train import load_model

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "metrics.json"


def main() -> None:
    X_train, X_test, y_train, y_test = load_processed(PROCESSED_DIR)
    model = load_model(MODEL_PATH)

    from src.evaluate import evaluate

    metrics_df = evaluate({"RandomForest": model}, X_test, y_test)
    metrics = metrics_df.iloc[0].to_dict()

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Метрики сохранены: {METRICS_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()

"""Этап сравнения трёх моделей"""
from src.compare_models import compare_models
from src.preprocessing import load_processed

PROCESSED_DIR = "data/processed"
OUTPUT_PATH = "models/comparison_table.csv"


def main() -> None:
    """Запускает сравнение трёх моделей и сохраняет результаты"""
    x_train, x_test, y_train, y_test = load_processed(PROCESSED_DIR)

    print("Сравнение трёх моделей")
    df_results = compare_models(x_train, y_train, x_test, y_test)

    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nРезультаты сохранены: {OUTPUT_PATH}")
    print("\n" + df_results.to_string(index=False))
    print("\nФинальная модель проекта: Random Forest")


if __name__ == "__main__":
    main()

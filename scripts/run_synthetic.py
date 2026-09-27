"""Шаг DVC-пайплайна: data/heart.csv -> data/heart_synth.csv.

Запуск (из корня heart_project): python -m scripts.run_synthetic
"""

from src.synthetic import generate_synthetic_data

INPUT_PATH = "data/heart.csv"
OUTPUT_PATH = "data/heart_synth.csv"


def main() -> None:
    df = generate_synthetic_data(INPUT_PATH, OUTPUT_PATH)
    print(f"Синтетика готова: {OUTPUT_PATH} ({len(df)} строк)")


if __name__ == "__main__":
    main()

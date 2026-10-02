"""Шаг DVC-пайплайна: data/heart_synth.csv -> data/processed/*.csv."""

from src.preprocessing import prepare_data

INPUT_PATH = "data/heart_synth.csv"
SAVE_DIR = "data/processed"


def main() -> None:
    X_train, X_test, *_ = prepare_data(INPUT_PATH, save_dir=SAVE_DIR)
    print(f"Предобработка готова: {SAVE_DIR} (train={X_train.shape}, test={X_test.shape})")


if __name__ == "__main__":
    main()

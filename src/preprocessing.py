"""
Предобработка данных
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Колонки, где 0 невозможен
ZERO_AS_MISSING_COLS = ["Cholesterol", "RestingBP"]

# Числовые колонки, которые масштабируем
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Категориальные колонки для one-hot кодирования
ONE_HOT_COLS = ["ChestPainType", "RestingECG"]

TARGET_COL = "HeartDisease"


def compute_medians(X_train: pd.DataFrame) -> dict:
    """Считает медианы по train, игнорируя нули"""
    return {col: X_train[col].replace(0, np.nan).median() for col in ZERO_AS_MISSING_COLS}


def clean(part: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Заменяет скрытые пропуски (нули) медианой и обрезает Oldpeak < 0"""
    part = part.copy()
    for col in ZERO_AS_MISSING_COLS:
        part[col] = part[col].replace(0, np.nan).fillna(medians[col])
    part["Oldpeak"] = part["Oldpeak"].clip(lower=0)
    return part


def encode(part: pd.DataFrame) -> pd.DataFrame:
    """Кодирует категориальные признаки"""
    part = part.copy()
    part["Sex"] = (part["Sex"] == "M").astype(int)
    part["ExerciseAngina"] = (part["ExerciseAngina"] == "Y").astype(int)
    part["ST_Slope"] = part["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})
    part = pd.get_dummies(part, columns=ONE_HOT_COLS, drop_first=True, dtype=int)
    return part


def prepare_data(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str | None = None,
):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    # Делим ДО очистки - иначе медианы/scaler "подсмотрят" тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    medians = compute_medians(X_train)
    X_train = clean(X_train, medians)
    X_test = clean(X_test, medians)

    X_train = encode(X_train)
    X_test = encode(X_test)
    # Редкая категория могла не попасть в test - выравниваем колонки
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    if save_dir is not None:
        save_processed(X_train, X_test, y_train, y_test, medians, scaler, save_dir)

    return X_train, X_test, y_train, y_test, medians, scaler


def save_processed(X_train, X_test, y_train, y_test, medians, scaler, save_dir: str):
    """Сохраняет результат prepare_data на диск"""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)

    pd.Series(medians, name="median").to_csv(out / "medians.csv")
    pd.DataFrame({"mean": scaler.mean_, "scale": scaler.scale_}, index=NUMERIC_COLS).to_csv(out / "scaler.csv")


def load_processed(save_dir: str):
    """Читает уже сохранённые файлы"""
    p = Path(save_dir)
    X_train = pd.read_csv(p / "X_train.csv")
    X_test = pd.read_csv(p / "X_test.csv")
    y_train = pd.read_csv(p / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(p / "y_test.csv")[TARGET_COL]
    return X_train, X_test, y_train, y_test

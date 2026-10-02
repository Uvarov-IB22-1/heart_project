"""
Тесты для src/preprocessing.py.

Запуск: pytest tests/ -v   (из корня heart_project)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean, compute_medians, encode, prepare_data


@pytest.fixture
def raw_part() -> pd.DataFrame:
    """Реалистичный кусок данных с проблемами, которые
    должна чинить clean(): нули в Cholesterol/RestingBP (скрытые
    пропуски) и отрицательный Oldpeak (ошибка измерения)."""
    return pd.DataFrame(
        {
            "Age": [40, 49, 37, 54],
            "Sex": ["M", "F", "M", "M"],
            "ChestPainType": ["ATA", "NAP", "ATA", "ASY"],
            "RestingBP": [140, 0, 130, 150],  # один явный 0
            "Cholesterol": [289, 180, 0, 195],  # один явный 0
            "FastingBS": [0, 0, 0, 1],
            "RestingECG": ["Normal", "Normal", "ST", "Normal"],
            "MaxHR": [172, 156, 98, 122],
            "ExerciseAngina": ["N", "N", "N", "Y"],
            "Oldpeak": [0.0, 1.0, -2.6, 1.5],  # одно отрицательное значение
            "ST_Slope": ["Up", "Flat", "Up", "Flat"],
        }
    )


@pytest.fixture
def medians(raw_part) -> dict:
    return compute_medians(raw_part)


def test_compute_medians_ignores_zeros(raw_part):
    """Нули не должны попадать в расчёт медианы - иначе медиана будет
    занижена самими же пропусками, которые она призвана исправлять."""
    medians = compute_medians(raw_part)
    # RestingBP без нуля: [140, 130, 150] -> медиана 140
    assert medians["RestingBP"] == 140
    # Cholesterol без нуля: [289, 180, 195] -> медиана 195
    assert medians["Cholesterol"] == 195


# ---------- clean ----------


def test_clean_removes_zeros(raw_part, medians):
    """После clean() в Cholesterol/RestingBP не должно остаться нулей -
    это и есть контракт функции, независимо от того, какая именно
    медиана была посчитана."""
    result = clean(raw_part, medians)
    assert (result["Cholesterol"] != 0).all()
    assert (result["RestingBP"] != 0).all()


def test_clean_fills_zero_with_train_median(raw_part, medians):
    """Проверяем, что подставляется именно заранее переданная медиана train."""
    result = clean(raw_part, medians)
    # строка 1: RestingBP был 0 -> должен стать medians['RestingBP']
    assert result.loc[1, "RestingBP"] == medians["RestingBP"]
    # строка 2: Cholesterol был 0 -> должен стать medians['Cholesterol']
    assert result.loc[2, "Cholesterol"] == medians["Cholesterol"]


def test_clean_clips_negative_oldpeak(raw_part, medians):
    """Oldpeak < 0 физически невозможен -> после clean() должен быть 0."""
    result = clean(raw_part, medians)
    assert (result["Oldpeak"] >= 0).all()
    assert result.loc[2, "Oldpeak"] == 0


def test_clean_does_not_mutate_input(raw_part, medians):
    """clean() должна возвращать копию."""
    original = raw_part.copy()
    clean(raw_part, medians)
    pd.testing.assert_frame_equal(raw_part, original)


# encode


def test_encode_produces_only_numeric_columns(raw_part, medians):
    """encode(): на выходе не должно остаться object/str-колонок"""
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert result.select_dtypes(exclude="number").empty


def test_encode_binary_columns(raw_part, medians):
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert set(result["Sex"].unique()) <= {0, 1}
    assert set(result["ExerciseAngina"].unique()) <= {0, 1}


def test_encode_ordinal_st_slope(raw_part, medians):
    """ST_Slope - порядковый признак: Up=0, Flat=1, Down=2."""
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert result.loc[0, "ST_Slope"] == 0  # было 'Up'
    assert result.loc[1, "ST_Slope"] == 1  # было 'Flat'


# ---------- prepare_data (полный пайплайн) ----------


def test_prepare_data_no_nan(tmp_path):
    """После полного пайплайна не должно остаться NaN - ни в train, ни в test"""
    csv_path = _make_synthetic_csv(tmp_path, n=200)
    X_train, X_test, y_train, y_test, medians, scaler = prepare_data(str(csv_path))
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_prepare_data_train_test_same_columns(tmp_path):
    """Колонки test должны точно совпадать с train"""
    csv_path = _make_synthetic_csv(tmp_path, n=200)
    X_train, X_test, *_ = prepare_data(str(csv_path))
    assert list(X_train.columns) == list(X_test.columns)


def test_prepare_data_clean_step_clips_oldpeak(tmp_path):
    """Oldpeak >= 0 - контракт clean(), а не prepare_data() целиком:
    после StandardScaler внутри prepare_data() Oldpeak центрируется
    к среднему и закономерно становится отрицательным у части строк.
    Поэтому здесь читаем CSV и проверяем clean() напрямую, до масштабирования.
    """
    df = pd.read_csv(_make_synthetic_csv(tmp_path, n=200))
    X = df.drop(columns="HeartDisease")
    medians = compute_medians(X)
    cleaned = clean(X, medians)
    assert (cleaned["Oldpeak"] >= 0).all()


def test_prepare_data_saves_processed_files(tmp_path):
    """save_dir должен создать все 6 файлов"""
    csv_path = _make_synthetic_csv(tmp_path, n=200)
    save_dir = tmp_path / "processed"
    prepare_data(str(csv_path), save_dir=str(save_dir))

    expected_files = {
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
        "medians.csv",
        "scaler.csv",
    }
    assert expected_files <= {p.name for p in save_dir.iterdir()}


def _make_synthetic_csv(tmp_path, n: int = 200) -> Path:
    """Генерирует небольшой валидный CSV с той же схемой, что и
    heart_synth.csv"""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Age": rng.integers(30, 80, n),
            "Sex": rng.choice(["M", "F"], n),
            "ChestPainType": rng.choice(["ATA", "NAP", "ASY", "TA"], n),
            "RestingBP": rng.integers(90, 180, n),
            "Cholesterol": rng.choice([0] + list(range(150, 350)), n),  # немного нулей
            "FastingBS": rng.integers(0, 2, n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], n),
            "MaxHR": rng.integers(80, 200, n),
            "ExerciseAngina": rng.choice(["N", "Y"], n),
            "Oldpeak": rng.uniform(-1, 4, n).round(1),  # с отрицательными
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
            "HeartDisease": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "synthetic_test_data.csv"
    df.to_csv(path, index=False)
    return path

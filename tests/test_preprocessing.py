"""
Тесты для src/preprocessing.py (Этап 3, п. 4.b методички).

Проверяем не "правильность с медицинской точки зрения", а КОНТРАКТ каждой
функции — то, что она гарантирует на выходе при любых допустимых входных
данных. Это и есть страховка DevOps-роли: если кто-то (в том числе сам
автор) случайно сломает логику при следующей правке src/preprocessing.py,
эти тесты упадут раньше, чем баг доедет до models.ipynb или до защиты проекта.

Запуск: pytest tests/ -v   (из корня heart_project)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean, compute_medians, encode, prepare_data


@pytest.fixture
def raw_part() -> pd.DataFrame:
    """Маленький, но реалистичный кусок данных с проблемами, которые
    и должна чинить clean(): нули в Cholesterol/RestingBP (скрытые
    пропуски) и отрицательный Oldpeak (ошибка измерения)."""
    return pd.DataFrame(
        {
            "Age": [40, 49, 37, 54],
            "Sex": ["M", "F", "M", "M"],
            "ChestPainType": ["ATA", "NAP", "ATA", "ASY"],
            "RestingBP": [140, 0, 130, 150],  # один явный нуль
            "Cholesterol": [289, 180, 0, 195],  # один явный нуль
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


# ---------- compute_medians ----------


def test_compute_medians_ignores_zeros(raw_part):
    """Нули не должны попадать в расчёт медианы — иначе медиана будет
    занижена самими же пропусками, которые она призвана исправлять."""
    medians = compute_medians(raw_part)
    # RestingBP без нуля: [140, 130, 150] -> медиана 140
    assert medians["RestingBP"] == 140
    # Cholesterol без нуля: [289, 180, 195] -> медиана 195
    assert medians["Cholesterol"] == 195


# ---------- clean ----------


def test_clean_removes_zeros(raw_part, medians):
    """После clean() в Cholesterol/RestingBP не должно остаться нулей —
    это и есть контракт функции, независимо от того, какая именно
    медиана была посчитана."""
    result = clean(raw_part, medians)
    assert (result["Cholesterol"] != 0).all()
    assert (result["RestingBP"] != 0).all()


def test_clean_fills_zero_with_train_median(raw_part, medians):
    """Проверяем не только "нет нулей", но и что подставляется именно
    заранее переданная медиана train, а не что-то ещё."""
    result = clean(raw_part, medians)
    # строка 1: RestingBP был 0 -> должен стать medians['RestingBP']
    assert result.loc[1, "RestingBP"] == medians["RestingBP"]
    # строка 2: Cholesterol был 0 -> должен стать medians['Cholesterol']
    assert result.loc[2, "Cholesterol"] == medians["Cholesterol"]


def test_clean_clips_negative_oldpeak(raw_part, medians):
    """Oldpeak < 0 физически невозможен -> после clean() должен быть 0."""
    result = clean(raw_part, medians)
    assert (result["Oldpeak"] >= 0).all()
    assert result.loc[2, "Oldpeak"] == 0  # было -2.6


def test_clean_does_not_mutate_input(raw_part, medians):
    """clean() должна возвращать копию, а не менять переданный df на
    месте — иначе повторный вызов на том же объекте даст неожиданный
    результат (частый источник багов в ноутбуках)."""
    original = raw_part.copy()
    clean(raw_part, medians)
    pd.testing.assert_frame_equal(raw_part, original)


# ---------- encode ----------


def test_encode_produces_only_numeric_columns(raw_part, medians):
    """Контракт encode(): на выходе не должно остаться object/str-колонок —
    иначе StandardScaler или модель упадут на следующем шаге."""
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert result.select_dtypes(exclude="number").empty


def test_encode_binary_columns(raw_part, medians):
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert set(result["Sex"].unique()) <= {0, 1}
    assert set(result["ExerciseAngina"].unique()) <= {0, 1}


def test_encode_ordinal_st_slope(raw_part, medians):
    """ST_Slope — порядковый признак: Up=0, Flat=1, Down=2."""
    cleaned = clean(raw_part, medians)
    result = encode(cleaned)
    assert result.loc[0, "ST_Slope"] == 0  # было 'Up'
    assert result.loc[1, "ST_Slope"] == 1  # было 'Flat'


# ---------- prepare_data (полный пайплайн) ----------


def test_prepare_data_no_nan(tmp_path):
    """После полного пайплайна не должно остаться NaN — ни в train,
    ни в test (главная проверка из clean.ipynb, п. 2.c)."""
    csv_path = _make_synthetic_csv(tmp_path, n=200)
    X_train, X_test, y_train, y_test, medians, scaler = prepare_data(str(csv_path))
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_prepare_data_train_test_same_columns(tmp_path):
    """Колонки test должны точно совпадать с train (важно, если в test
    случайно не попала редкая категория из one-hot кодирования)."""
    csv_path = _make_synthetic_csv(tmp_path, n=200)
    X_train, X_test, *_ = prepare_data(str(csv_path))
    assert list(X_train.columns) == list(X_test.columns)


def test_prepare_data_clean_step_clips_oldpeak(tmp_path):
    """Oldpeak >= 0 — контракт clean(), а не prepare_data() целиком:
    после StandardScaler внутри prepare_data() Oldpeak центрируется
    к среднему и закономерно становится отрицательным у части строк
    (это уже не "секунды ST-депрессии", а "число стандартных отклонений").
    Поэтому здесь читаем CSV и проверяем clean() напрямую, до масштабирования.
    """
    df = pd.read_csv(_make_synthetic_csv(tmp_path, n=200))
    X = df.drop(columns="HeartDisease")
    medians = compute_medians(X)
    cleaned = clean(X, medians)
    assert (cleaned["Oldpeak"] >= 0).all()


def test_prepare_data_saves_processed_files(tmp_path):
    """save_dir должен создать все 6 processed-файлов (X/y train/test +
    medians + scaler) — это то, что читает models.ipynb."""
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


# ---------- вспомогательное ----------


def _make_synthetic_csv(tmp_path, n: int = 200) -> Path:
    """Генерирует небольшой валидный CSV с той же схемой, что и
    heart_synth.csv, включая немного нулей/отрицательных Oldpeak —
    чтобы prepare_data() было на чём отработать очистку."""
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

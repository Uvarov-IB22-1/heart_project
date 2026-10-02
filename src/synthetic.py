"""
Генерация синтетических данных для heart_project.

Логика 1-в-1 повторяет notebooks/synthetic.ipynb: случайные
записи с повторами + 5% шум по числовым признакам + физические
ограничения (clip) на диапазоны значений - чтобы результат оставался
правдоподобным, а не просто зашумлённым.
"""

import numpy as np
import pandas as pd

NUMERIC_NOISE_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# (колонка, min, max) - границы, за которые значения не могут выходить
# физически, даже после добавления шума
CLIP_RANGES = {
    "Age": (20, 90),
    "RestingBP": (80, 200),
    "Cholesterol": (0, 600),
    "MaxHR": (60, 210),
    "Oldpeak": (0, 6),
}


def generate_synthetic_data(
    input_path: str,
    output_path: str,
    target_size: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Дополняет датасет до target_size строк через бутстрэп + шум.

    Сохраняет исходные строки как есть и добавляет к ним сгенерированные -
    оригинальные данные никогда не удаляются и не перезаписываются.
    """
    np.random.seed(seed)
    df = pd.read_csv(input_path)
    n_to_generate = target_size - len(df)

    if n_to_generate <= 0:
        df.to_csv(output_path, index=False)
        return df

    # 1. Берём случайные существующие записи
    boot_idx = np.random.randint(0, len(df), size=n_to_generate)
    synth = df.iloc[boot_idx].reset_index(drop=True)

    # запоминаем, где холестерин был нулём (скрытый пропуск) - шум не
    # должен превратить осмысленный "пропуск" в случайное число
    zero_chol = synth["Cholesterol"] == 0

    # 2. Добавляем шум 5% от std оригинального (не бутстрапнутого) df
    for col in NUMERIC_NOISE_COLS:
        std = df[col].std()
        noise = np.random.normal(0, 0.05 * std, size=n_to_generate)
        synth[col] = synth[col] + noise

    # 3. Возвращаем "пропуски" на место и обрезаем до физических границ
    synth.loc[zero_chol, "Cholesterol"] = 0
    for col, (lo, hi) in CLIP_RANGES.items():
        synth[col] = synth[col].clip(lo, hi).round(0 if col != "Oldpeak" else 1)
    for col in ["Age", "RestingBP", "Cholesterol", "MaxHR"]:
        synth[col] = synth[col].astype(int)

    synth["FastingBS"] = synth["FastingBS"].round(0).astype(int)
    synth["HeartDisease"] = synth["HeartDisease"].round(0).astype(int)

    df = pd.concat([df, synth], ignore_index=True)
    df.to_csv(output_path, index=False)
    return df

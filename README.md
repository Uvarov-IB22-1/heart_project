# Heart Project

Использованный датасет: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Инструкция по настройке окружения и началу работы над проектом.

## 1. Клонирование репозитория
Склонируйте проект и перейдите в его папку:
```bash
git clone https://github.com/Uvarov-IB22-1/heart_project.git
cd heart_project
```

## 2. Виртуальное окружение
Создайте и активируйте изолированную среду Python:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

## 3. Установка зависимостей
Установите все необходимые библиотеки из зафиксированного списка:
```bash
pip install -r requirements.txt
```

## 4. Полное воспроизведение проекта (данные → модель → метрики)

Весь путь — генерация синтетических данных, предобработка, обучение
и тюнинг модели, оценка на тесте — воспроизводится одной командой,
без запуска ноутбуков вручную:

```bash
dvc repro
```

DVC сам определит, какие шаги пайплайна (`dvc.yaml`) уже актуальны
(по хэшам файлов в `dvc.lock`), и пересчитает только то, что реально
изменилось — от `data/heart.csv` до `models/model.pkl` и `metrics.json`.

Посмотреть итоговые метрики модели:
```bash
dvc metrics show
```

Посмотреть граф пайплайна:
```bash
dvc dag
```

## 5. Структура проекта

```
data/heart.csv              исходный датасет (Kaggle, 918 строк)
data/heart_synth.csv        расширенный датасет (3000 строк) — генерируется dvc repro
data/processed/             X_train/X_test/y_train/y_test + medians/scaler — генерируется dvc repro
models/model.pkl            обученная модель — генерируется dvc repro
metrics.json                метрики финальной модели — генерируется dvc repro

src/                         переиспользуемая логика (импортируется и ноутбуками, и scripts/)
scripts/                     CLI-обёртки над src/ — то, что реально запускает dvc.yaml
tests/                       pytest-тесты для src/

notebooks/                   разведочный анализ (EDA), визуализации, обоснование решений —
                              для чтения и отчёта, а не для получения актуального результата
```

Если нужно как воспроизвести конкретно цифры/графики для отчёта — открой `notebooks/`.
Если нужно как **пересчитать** данные/модель — используй `dvc repro`, а не ноутбуки.

## 6. Настройка SSH (для коммитов без ввода пароля)
Если вы хотите использовать SSH вместо HTTPS:
1. Сгенерируйте ключ (если его еще нет): `ssh-keygen -t ed25519`
2. Скопируйте публичный ключ (`cat ~/.ssh/id_ed25519.pub`) и добавьте его в свой аккаунт GitHub (Settings -> SSH and GPG keys).
3. Переключите локальный репозиторий на SSH-адрес:
   ```bash
   git remote set-url origin git@github.com:Uvarov-IB22-1/heart_project.git
   ```

## 7. Воспроизводимость метрик

`RandomForest` и `LogisticRegression` воспроизводятся детерминированно (бит-в-бит)
на любой машине при одинаковой версии `requirements.txt`.

`CatBoost` и `GridSearchCV(n_jobs=-1)` в `models.ipynb` используют многопоточность:
из-за порядка операций с плавающей точкой при разном числе ядер CPU метрики
`CatBoost` и выбор "лучших" гиперпараметров `RandomForest` могут отличаться
в третьем знаке между машинами (например, `max_depth=15` вместо `max_depth=None`
при статистически неотличимом F1). На итоговое качество финальной модели
(F1 ≈ 0.99, ROC-AUC ≈ 0.998) это не влияет.

## 8. Рабочий процесс
**Перед началом работы:**
```bash
git pull origin main
```

**По завершении работы:**
```bash
git add .
git commit -m "Краткое и понятное описание изменений"
git push origin main
```
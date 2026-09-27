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

## 4. Работа с данными (DVC)
Для подгрузки версионированных данных (если они хранятся в DVC):
```bash
dvc pull
```

## 5. Настройка SSH (для коммитов без ввода пароля)
Если вы хотите использовать SSH вместо HTTPS:
1. Сгенерируйте ключ (если его еще нет): `ssh-keygen -t ed25519`
2. Скопируйте публичный ключ (`cat ~/.ssh/id_ed25519.pub`) и добавьте его в свой аккаунт GitHub (Settings -> SSH and GPG keys).
3. Переключите локальный репозиторий на SSH-адрес:
   ```bash
   git remote set-url origin git@github.com:Uvarov-IB22-1/heart_project.git
   ```

## 6. Воспроизводимость метрик

`RandomForest` и `LogisticRegression` воспроизводятся детерминированно (бит-в-бит)
на любой машине при одинаковой версии `requirements.txt`.

`CatBoost` и `GridSearchCV(n_jobs=-1)` в `models.ipynb` используют многопоточность:
из-за порядка операций с плавающей точкой при разном числе ядер CPU метрики
`CatBoost` и выбор "лучших" гиперпараметров `RandomForest` могут отличаться
в третьем знаке между машинами (например, `max_depth=15` вместо `max_depth=None`
при статистически неотличимом F1). На итоговое качество финальной модели
(F1 ≈ 0.99, ROC-AUC ≈ 0.998) это не влияет.

## 7. Рабочий процесс
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
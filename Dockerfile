# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY iris_logreg_docker.py ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir scikit-learn

# Указываем команду для запуска скрипта
CMD ["python", "iris_logreg_docker.py"]

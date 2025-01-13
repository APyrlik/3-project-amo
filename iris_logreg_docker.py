import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Загрузка датасета ирисов
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Сохранение модели в файл
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Тестовое предсказание
test_input = np.array([[1, 1, 1, 1]])
prediction = model.predict(test_input)
print(f"Prediction for test input {test_input}: {prediction[0]}")

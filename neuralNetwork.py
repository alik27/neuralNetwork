from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Загружаем набор данных из keras
(xtr, ytr), (x_test, y_test) = mnist.load_data()

# Классификация на классы
classes = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
'''
# Вывод 50 картинок
plt.figure(figsize=(10,10))
for i in range(100,150):
    plt.subplot(5,10,i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xtr[i], cmap=plt.cm.binary)
    plt.xlabel(classes[ytr[i]])
'''
# Преобразование размерности
xtr = xtr.reshape(60000, 784)
# Нормализация данных
xtr = xtr / 255
# Преобразовываем метки
ytr = utils.to_categorical(ytr, 10)

# Создаем последовательную модель
model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Компилируем
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Обучение
training = model.fit(xtr, ytr, 
                    batch_size=200, 
                    epochs=10,  
                    verbose=1)

# Распознавание
predict = model.predict(xtr)

n = 4
plt.imshow(xtr[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# Результат
print('Вероятности: ',predict[n])
print('Вероятный вариант: ',classes[np.argmax(predict[n])],'(',np.argmax(predict[n]),')',)
print('Правильный вариант: ',classes[np.argmax(ytr[n])],'(',np.argmax(ytr[n]),')',)

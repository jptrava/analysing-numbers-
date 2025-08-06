import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Flatten(input_shape=(28, 28)),  # transforma imagem 28x28 em vetor
    Dense(128, activation='relu'),  # camada oculta com 128 neurônios
    Dense(10, activation='softmax')  # saída com 10 neurônios (classes 0-9)
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {accuracy * 100:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
import random


index = random.randint(1, 9999)

# Mostrar a imagem
plt.imshow(x_test[index], cmap='gray')
plt.title("Imagem de Teste")
plt.axis('off')
plt.show()


prediction = model.predict(np.expand_dims(x_test[index], axis=0))  # Adiciona dimensão para batch
predicted_label = np.argmax(prediction)

print(f"✅ A rede previu que este número é: {predicted_label}")

#http://python-programmieren.maximilianwittmann.de/kunstliche-intelligenz-programmieren/
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from PIL import Image
import numpy as np

# Laden Sie den MNIST-Datensatz herunter
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Wir müssen die Bilder neu formatieren, damit sie mit Keras funktionieren
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalisieren Sie die Bilder auf Werte zwischen 0 und 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Konvertieren Sie die Labels in kategorische Daten
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Erstellen Sie das Modell
model = Sequential()

# Fügen Sie das erste Convolutional Layer hinzu
model.add(Flatten(input_shape=(28, 28, 1)))

# Fügen Sie ein MaxPooling Layer hinzu
model.add(Dense(64, activation='relu'))

# Fügen Sie das Output Layer hinzu. Da wir MNIST verwenden, haben wir 10 Nodes.
model.add(Dense(10, activation='softmax'))

# Kompilieren Sie das Modell
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trainieren Sie das Modell
model.fit(train_images, train_labels, epochs=50, batch_size=64)

# Evaluieren Sie das Modell
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# Test accuracy konvergiert gegen 1.000
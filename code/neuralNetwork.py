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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Fügen Sie ein MaxPooling Layer hinzu
model.add(MaxPooling2D((2, 2)))

# Fügen Sie ein weiteres Convolutional Layer hinzu
model.add(Conv2D(64, (3, 3), activation='relu'))

# Fügen Sie ein weiteres MaxPooling Layer hinzu
model.add(MaxPooling2D((2, 2)))

# Fügen Sie ein Flatten Layer hinzu, um die Daten für die Fully Connected Layers vorzubereiten
model.add(Flatten())

# Fügen Sie ein Fully Connected Layer mit 64 Nodes hinzu
model.add(Dense(64, activation='relu'))

# Fügen Sie das Output Layer hinzu. Da wir MNIST verwenden, haben wir 10 Nodes.
model.add(Dense(10, activation='softmax'))

# Kompilieren Sie das Modell
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trainieren Sie das Modell
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluieren Sie das Modell
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# Bild öffnen und in Graustufen umwandeln
img = Image.open('test_input.png').convert('L')

# Bild in ein Numpy-Array umwandeln und skalieren
img = np.resize(img, (28, 28, 1))
im2arr = np.array(img)
im2arr = im2arr.reshape(1, 28, 28, 1)

# Vorhersage mit dem Modell machen
pred = model.predict(im2arr)
predicted_class = np.argmax(pred)
print(predicted_class)
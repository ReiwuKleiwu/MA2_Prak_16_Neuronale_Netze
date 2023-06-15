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

# ändere Label zu 1, wenn 5 sonst 0
train_labels = np.where(train_labels == 5, 1, 0)
test_labels = np.where(test_labels == 5, 1, 0)

# reserviere 20 Bilder zum manuellen Testen
manualTestImages = test_images[-20:]
manualTestLabels = test_labels[-20:]

# Erstellen Sie das Modell
model = Sequential()
# Bild in Vektor umwandeln und Input Layer mit 28*28*1 Neuronen erstellen
model.add(Flatten(input_shape=(28, 28, 1)))
# Output Layer mit einem Neuron erstellen
model.add(Dense(1, activation='sigmoid'))
# Kompilieren Sie das Modell
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Trainieren Sie das Modell
model.fit(train_images, train_labels, epochs=25, batch_size=64)
# Evaluieren Sie das Modell
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)



# Manuelles Testen
for (tImg, tLabel) in zip(manualTestImages, manualTestLabels):
  tImg = tImg.reshape(1, 28, 28, 1)
  pred = model.predict(tImg)
  print("Expected: " + str(tLabel) + "; Got: " + str(np.round(pred)))

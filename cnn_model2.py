# Importar las bibliotecas necesarias
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.saving import save_model
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Inicialización de la CNN
classifier = Sequential()

# Paso 1 - Primera capa de convolución y pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 2 - Segunda capa de convolución y pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 3 - Tercera capa de convolución y pooling
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 4 - Aplanamiento (Flatten)
classifier.add(Flatten())

# Paso 5 - Conexión completa (Full connection)
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))

# Modificación: Cambiamos a 27 neuronas para las 26 letras + "OK"
classifier.add(Dense(27, activation='softmax'))

# Compilar la CNN
classifier.compile(
    optimizer=optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Generación de datos de entrenamiento y prueba
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar los datasets de entrenamiento y prueba
training_set = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Entrenar el modelo
model = classifier.fit(
    training_set,
    steps_per_epoch=100,
    epochs=10,
    validation_data=test_set,
    validation_steps=25)


# Saving the model
save_model(classifier, 'Trained_model.keras')


# Graficar los resultados del entrenamiento
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

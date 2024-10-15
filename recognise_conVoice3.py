import cv2
import numpy as np
import pyttsx3
import time
from keras.models import load_model

# Inicializa el motor de TTS
engine = pyttsx3.init()

# Mostrar dispositivos de audio disponibles
voices = engine.getProperty('voices')
virtual_device_index = 1  # Cambia según tu configuración
engine.setProperty('voice', voices[virtual_device_index].id)

def speak(text):
    """Función para hablar usando TTS."""
    engine.say(text)
    engine.runAndWait()

image_x, image_y = 64, 64
classifier = load_model('Trained_model.keras')

def predictor(image_path):
    """Función para predecir la letra o gesto desde una imagen."""
    from keras.preprocessing import image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    class_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZo'  # Letras + "OK"
    predicted_index = np.argmax(result[0])
    predicted_label = class_labels[predicted_index]
    
    return predicted_label

def nothing(x):
    pass

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 12, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 47, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 21, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Variables para lógica de reconocimiento y confirmación
img_text = ""
confirmed_text = ""
last_prediction = ""
start_time = None  # Para medir el tiempo de estabilidad

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Obtener los valores de los trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Dibujar un cuadro para la detección de gestos
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Guardar y predecir la imagen
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    
    current_prediction = predictor(img_name)  # Predecir el gesto actual

    # Mostrar la letra predicha en la ventana principal
    cv2.putText(frame, f"Predicted: {current_prediction}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Text: {confirmed_text}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # Lógica para confirmar si el gesto se mantiene estable por 5 segundos
    if current_prediction == last_prediction:
        if start_time is None:
            start_time = time.time()  # Inicia el temporizador
        elif time.time() - start_time >= 5:  # Si pasan 5 segundos
            confirmed_text += current_prediction  # Agregar letra confirmada
            print(f"Confirmed: {confirmed_text}")
            start_time = None  # Reiniciar el temporizador
    else:
        start_time = None  # Reiniciar si el gesto cambia

    last_prediction = current_prediction  # Actualizar el último gesto

    # Mostrar las ventanas
    cv2.imshow("test", frame)

    # Gesto especial para reproducir el texto
    if current_prediction == "o" and start_time is None:
        print(f"Speaking: {confirmed_text}")
        speak(confirmed_text)
        confirmed_text = ""  # Reiniciar el texto después de hablar

    if cv2.waitKey(1) == 27:  # Presionar ESC para salir
        break

cam.release()
cv2.destroyAllWindows()

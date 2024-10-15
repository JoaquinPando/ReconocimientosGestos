import cv2
import numpy as np
import pyttsx3
from keras.models import load_model

# Inicializa el motor de TTS
engine = pyttsx3.init()

# Mostrar dispositivos de audio disponibles
voices = engine.getProperty('voices')
print("Dispositivos disponibles:")
for index, voice in enumerate(voices):
    print(f"{index}: {voice.name}")

# Selecciona el dispositivo (por ejemplo, el driver virtual)
virtual_device_index = 1  # Cambia este número según el driver que quieras usar
engine.setProperty('voice', voices[virtual_device_index].id)

def nothing(x):
    pass

image_x, image_y = 64, 64
classifier = load_model('Trained_model.h5')

def predictor(image_path):
    from keras.preprocessing import image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    
    # Definir los resultados de las clases
    class_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    predicted_index = np.argmax(result[0])
    predicted_label = class_labels[predicted_index]
    
    return predicted_label

def speak(text):
    engine.say(text)
    engine.runAndWait()

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 12, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 47, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 21, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

img_counter = 0
img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    
    # Guarda la imagen procesada
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    
    img_text = predictor(img_name)  # Predecir el gesto
    if img_text:  # Si se reconoce un gesto
        print("Predicted Sign: ", img_text)
        speak(img_text)  # Reproducir el gesto detectado
        
    if cv2.waitKey(1) == 27:  # Esc para salir
        break

cam.release()
cv2.destroyAllWindows()

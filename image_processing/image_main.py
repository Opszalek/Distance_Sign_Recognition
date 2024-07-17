from utils.utils import timeit
from image_processing.sign_recognition import SignRecognition
from ultralytics import YOLO
import cv2
import pytesseract
from image_processing.text_detection import TextDetection
from image_processing.text_recognition import TextRecognition
import os
import sys
# Variables and paths
path_to_model = '../Models/Sign_recognition/model_1.pt'
path_to_save_cropped = '../TESTS/cropped'
test_image = False
path_to_video = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/Videos/LUCID_TRI050S-C_241401047__20240522004730540_video9.mp4'
path_to_image = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/znaki/img_4.png'
# Initialize models
sign_rec = SignRecognition(path_to_model, path_to_save_cropped, show_images=True, save_cropped=True)
text_rec = TextDetection()
text_det = TextRecognition()
# cap = cv2.VideoCapture(0)#Kamera
cap = cv2.VideoCapture(path_to_video)  #Filmik


model = YOLO(path_to_model)

path='/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/images'
while True:
    if not test_image:
        ret, image = cap.read()
        if not ret:
            break
# for picture in os.listdir(path):
#     image = cv2.imread(path + '/'+picture)

    signs, results = sign_rec.process_image(image, show_signs=False)
    text_images = text_rec.detect_handler(signs)
    numer2 = 0
    sign_text = text_det.predict_handler(text_images)
    print(sign_text)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

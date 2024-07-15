from utils.utils import timeit
from image_processing.sign_recognition import SignRecognition
from ultralytics import YOLO
import cv2
import pytesseract
from image_processing.text_detection import TextDetection
from image_processing.text_recognition import TextRecognition
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


# Load test image
# image = cv2.imread('../TESTS/img.png')  #load test image so we can test without camera


# @timeit # decorator to measure time

# -----------------Functions---from---OCR----------------#
def ocr_text(image):
    return pytesseract.image_to_string(image)


model = YOLO(path_to_model)
@timeit
def test_main():
    imagero = cv2.imread(path_to_image)
    signs = sign_rec.process_image(imagero)
    cv2.imshow('im14age', signs[0][0])
    text_images = text_rec.detect_text(signs[0][0])
    print(text_images)
    # cv2.imshow('ima12ge', text_images[0])
    if text_images:
        cv2.imshow('ima325ge', text_images[0])
        sign_text = text_det.predict_text(text_images[0])
        print(sign_text)
    # cv2.imshow('image', signs[0])
    # cv2.waitKey(0)
test_main()
sys.exit()
while True:
    if not test_image:
        ret, image = cap.read()
        if not ret:
            break
    # results = model.track(image, persist=True, tracker='bytetrack.yaml')
    #
    # # Visualize the results on the frame
    # annotated_frame = results[0].plot()
    #
    # # Display the annotated frame
    # annotated_frame = cv2.resize(annotated_frame, (640, 640))
    # cv2.imshow("YOLOv8 Tracking", annotated_frame)
    #
    # # Break the loop if 'q' is pressed
    # cv2.waitKey(0)
    imagero = cv2.imread(path_to_image)
    signs = sign_rec.process_image(imagero)
    cv2.imshow('im14age', signs[0][0])
    text_images = text_rec.detect_text(signs[0][0])
    print(text_images)
    # cv2.imshow('ima12ge', text_images[0])
    if text_images:
        cv2.imshow('ima325ge', text_images[0])
        sign_text = text_det.predict_text(text_images[0])
        print(sign_text)
    # cv2.imshow('image', signs[0])
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

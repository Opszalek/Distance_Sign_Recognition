from utils.utils import timeit
from image_processing.sign_recognition import SignRecognition
from ultralytics import YOLO
import cv2
import pytesseract

# Variables and paths
path_to_model = '../Models/Sign_recognition/model_1.pt'
path_to_save_cropped = '../TESTS/cropped'
test_image = False
path_to_video = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/VIDIO/VID_20240416_235323.mp4'
# Initialize model and camera
sign_rec = SignRecognition(path_to_model, path_to_save_cropped, show_images=True, save_cropped=True)
# cap = cv2.VideoCapture(0)#Kamera
cap = cv2.VideoCapture(path_to_video)  #Filmik

# Load test image
image = cv2.imread('../TESTS/img.png')  #load test image so we can test without camera


# @timeit # decorator to measure time

# -----------------Functions---from---OCR----------------#
def ocr_text(image):
    return pytesseract.image_to_string(image)

model = YOLO(path_to_model)
while True:
    if not test_image:
        ret, image = cap.read()
        if not ret:
            break
    results = model.track(image, persist=True, tracker='bytetrack.yaml')

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    annotated_frame = cv2.resize(annotated_frame, (640, 640))
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    cv2.waitKey(0)



        # signs = sign_rec.process_image(image)

cap.release()
cv2.destroyAllWindows()

from utils.utils import timeit
from image_processing.sign_recognition import SignRecognition
from ultralytics import YOLO
import cv2
import pytesseract

# Variables and paths
path_to_model = '../Models/model_1.pt'
path_to_save_cropped = '../TESTS/cropped'
test_image = True

# Initialize model and camera
sign_rec = SignRecognition(path_to_model, path_to_save_cropped)
cam = cv2.VideoCapture(0)

# Load test image
image = cv2.imread('../TESTS/img.png')  #load test image so we can test without camera

# @timeit # decorator to measure time

# -----------------Functions---from---OCR----------------#
def ocr_text(image):
    return pytesseract.image_to_string(image)


while True:
    if not test_image:
        ret, image = cam.read()
        if not ret:
            break

    results = sign_rec.predict_sign(image)
    bboxes = sign_rec.return_bboxes(results)
    signs = sign_rec.crop_signs(image, bboxes)
    sign_rec.save_cropped(signs)
    print(ocr_text(signs[0][0]))

cam.release()
cv2.destroyAllWindows()

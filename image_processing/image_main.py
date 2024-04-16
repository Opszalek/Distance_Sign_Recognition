from utils.utils import timeit
from ultralytics import YOLO
import cv2
import pytesseract

model = YOLO("../Models/model_1.pt")  # load a pretrained model (recommended for training)

cam = cv2.VideoCapture(0)
image = cv2.imread('../TESTS/img.png')  #load test image so we can test without camera
i = 0
test_image = True


# -----------------Functions---from---object_detection----------------#
# @timeit # decorator to measure time
def predict_sign(data):
    return model.predict(source=data)[0]


def return_bboxes(results):
    bboxes = []
    for detection in results.boxes.data.tolist():
        x, y, w, h, score, class_id = detection
        bboxes.append((x, y, w, h, score, class_id))
    return bboxes


def crop_signs(image, bboxes):
    signs = []
    for box in bboxes:
        x, y, w, h, score, class_id = box
        crop = image[int(y):int(h), int(x):int(w)]
        signs.append([crop, score, class_id])
    return signs


def save_cropped(signs, gps=None):
    # TODO: Change path to be class variable
    for sign in signs:
        crop, score, class_id = sign
        score = round(score, 2)
        if gps:
            cv2.imwrite(f'../TESTS/cropped/crop_sc-{score}_cl-{class_id}_gps{-gps}.png', crop)
        else:
            cv2.imwrite(f'../TESTS/cropped/crop_sc-{score}_cl-{class_id}.png', crop)


# -----------------Functions---from---OCR----------------#
def ocr_text(image):
    return pytesseract.image_to_string(image)


while True:
    if not test_image:
        ret, image = cam.read()
        if not ret:
            break

    results = predict_sign(image)
    bboxes = return_bboxes(results)
    signs = crop_signs(image, bboxes)
    save_cropped(signs)
    print(ocr_text(signs[0][0]))

cam.release()
cv2.destroyAllWindows()

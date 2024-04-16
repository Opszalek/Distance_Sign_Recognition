from ultralytics import YOLO
import cv2


class SignRecognition(YOLO):
    def __init__(self, model_path, path_to_save_cropped='../TESTS/cropped'):
        super().__init__(model_path)
        self.path_to_save_cropped = path_to_save_cropped

    def predict_sign(self, data):
        return self.predict(source=data)[0]

    def save_cropped(self, signs, gps=None):
        for sign in signs:
            crop, score, class_id = sign
            score = round(score, 2)
            if gps:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}_gps{-gps}.png', crop)
            else:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}.png', crop)

    @staticmethod
    def crop_signs(image, bboxes):
        signs = []
        for box in bboxes:
            x, y, w, h, score, class_id = box
            crop = image[int(y):int(h), int(x):int(w)]
            signs.append([crop, score, class_id])
        return signs

    @staticmethod
    def return_bboxes(results):
        bboxes = []
        for detection in results.boxes.data.tolist():
            x, y, w, h, score, class_id = detection
            bboxes.append((x, y, w, h, score, class_id))
        return bboxes

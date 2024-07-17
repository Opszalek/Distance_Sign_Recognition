import numpy as np
from ultralytics import YOLO
import cv2


class SignRecognition(YOLO):
    def __init__(self, model_path, path_to_save_cropped='../TESTS/cropped', **kwargs):
        super().__init__(model_path)
        self.path_to_save_cropped = path_to_save_cropped

        self.path_to_save_images = kwargs.get('path_to_save_images', '../TESTS/images')
        self.show_images = kwargs.get('show_images', False)
        self.save_images = kwargs.get('save_images', False)
        self.save_cropped = kwargs.get('save_cropped', False)
        self.show_signs = kwargs.get('show_signs', False)

    def predict_sign(self, data):
        return self.predict(source=data)[0]

    def save_cropped_signs(self, signs, results, gps=None):
        for sign, result in zip(signs, results):
            crop, (score, class_id) = sign, result
            score = round(score, 2)
            if gps:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}_gps-{gps}.png', crop)
            else:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}.png', crop)

    def save_image(self, image):  # TODO: Add unique name for each image
        cv2.imwrite(f'{self.path_to_save_images}/image.png', image)

    @staticmethod
    def show_image(image, bboxes):
        image_ = image.copy()
        for box in bboxes:
            x, y, w, h, score, class_id = box
            cv2.rectangle(image_, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(image_, f'{round(score, 2)} {class_id}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 5)
            cv2.rectangle(image_, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
        image_ = cv2.resize(image_, (640, 640))
        cv2.imshow('image', image_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    @staticmethod
    def crop_signs(image, bboxes):
        signs = []
        results = []
        for box in bboxes:
            x, y, w, h, score, class_id = box
            crop = image[int(y):int(h), int(x):int(w)]
            signs.append(crop)
            results.append([score, class_id])
        return signs, results

    @staticmethod
    def return_bboxes(results):
        bboxes = []
        for detection in results.boxes.data.tolist():
            x, y, w, h, score, class_id = detection
            bboxes.append((x, y, w, h, score, class_id))
        return bboxes

    def args_handler(self, image, bboxes, signs, results):
        if self.save_images:
            self.save_image(image)
        if self.save_cropped:
            self.save_cropped_signs(signs, results)
        if self.show_images:
            self.show_image(image, bboxes)
        if self.show_signs:
            self.show_cropped_signs(signs)

    @staticmethod
    def show_cropped_signs(signs):
        numer = 0
        for sign in signs:
            numer += 1
            cv2.imshow(f'Sign: {numer}', sign)

    def process_image(self, image, show_signs=False):
        """
        Process image and return cropped signs
        :param image: image to process
        :param show_signs: if True show cropped signs
        :return: list of cropped signs
        """
        self.show_signs = show_signs
        output = self.predict_sign(image)
        bboxes = self.return_bboxes(output)
        signs, results = self.crop_signs(image, bboxes)
        self.args_handler(image, bboxes, signs, results)
        return signs, results

from ultralytics import YOLO
import cv2
from image_processing import sign_tracker


class SignRecognition(YOLO):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path)
        self.show_images = kwargs.get('show_images', False)
        self.show_signs = kwargs.get('show_signs', False)

    def predict_sign(self, data):
        return self.predict(source=data)[0]

    @staticmethod
    def show_image(image, bboxes):
        image_ = image.copy()
        for box in bboxes:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_, f'{round(score, 2)} {class_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 5)
            cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        image_ = cv2.resize(image_, (640, 640))
        cv2.imshow('image', image_)

    @staticmethod
    def crop_signs(image, bboxes):
        signs = []
        results = []
        for box in bboxes:
            x1, y1, x2, y2, score, class_id = box
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            signs.append(crop)
            results.append(box)
        return signs, results

    @staticmethod
    def return_bboxes(results):
        return results.boxes.data.tolist()

    def args_handler(self, image, bboxes, signs, results):
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

    def process_image(self, image):
        """
        Process image and return cropped signs
        :param image: image to process
        :return: list of cropped signs and results
        """
        output = self.predict_sign(image)
        bboxes = self.return_bboxes(output)
        signs, results = self.crop_signs(image, bboxes)
        self.args_handler(image, bboxes, signs, results)
        return signs, results

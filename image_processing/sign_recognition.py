from ultralytics import YOLO
import cv2
from image_processing import sign_tracker


class SignRecognition(YOLO):
    def __init__(self, model_path, path_to_save_cropped='../TESTS/cropped', **kwargs):
        #tracker
        self.tracker = sign_tracker.SignTracker()
        super().__init__(model_path)
        self.path_to_save_cropped = path_to_save_cropped
        self.sign_number = 250  #TODO:create number reader from dict
        self.path_to_save_images = kwargs.get('path_to_save_images', '../TESTS/images')
        self.show_images = kwargs.get('show_images', False)
        self.save_images = kwargs.get('save_images', False)
        self.save_cropped = kwargs.get('save_cropped', False)
        self.show_signs = kwargs.get('show_signs', False)

    def predict_sign(self, data):
        return self.predict(source=data)[0]

    def save_cropped_signs(self, signs, results, gps=None):
        for sign, result in zip(signs, results):
            crop, (x1, y1, x2, y2, score, class_id) = sign, result
            print("stop2")
            print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
            score = round(score, 2)
            if x1 > 2048 - 700:  #TODO: delete it, just for testing
                if gps:
                    cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}_gps-{gps}.png', crop)
                else:
                    cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-nr{self.sign_number}.png', crop)
                    #add blur to image
                    crop = cv2.GaussianBlur(crop, (7, 7), 0)
                    cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-nr{self.sign_number}_blurred.png', crop)

                self.sign_number += 1

    def save_image(self, image):  # TODO: Add unique name for each image
        cv2.imwrite(f'{self.path_to_save_images}/image.png', image)

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
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

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
        selected_signs, selected_results = self.tracker.handle_tracking(list(zip(signs, results)))
        if len(selected_signs) > 0:
            cv2.imshow('image_', selected_signs[0])
            print(selected_results[0])
            cv2.waitKey(0)
        self.tracker.draw_bboxes(image)
        self.args_handler(image, bboxes, selected_signs, selected_results)
        return selected_signs, selected_results

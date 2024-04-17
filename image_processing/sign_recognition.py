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

    def predict_sign(self, data):
        return self.predict(source=data)[0]

    def save_cropped_signs(self, signs, gps=None):
        for sign in signs:
            crop, score, class_id = sign
            score = round(score, 2)
            if gps:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}_gps-{gps}.png', crop)
            else:
                cv2.imwrite(f'{self.path_to_save_cropped}/sign_sc-{score}_cl-{class_id}.png', crop)

    def save_image(self, image):  # TODO: Add unique name for each image
        cv2.imwrite(f'{self.path_to_save_images}/image.png', image)

    @staticmethod
    def show_image(image, bboxes):
        for box in bboxes:
            x, y, w, h, score, class_id = box
            cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(image, f'{round(score, 2)} {class_id}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255), 5)
            cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
        image = cv2.resize(image, (640, 640))
        cv2.imshow('image', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

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

    def args_handler(self, image, bboxes, signs):
        if self.save_images:
            self.save_image(image)
        if self.save_cropped:
            self.save_cropped_signs(signs)
        if self.show_images:
            self.show_image(image, bboxes)

    def process_image(self, image):
        """
        Process image and return cropped signs
        :param image: image to process
        :return: list of cropped signs
        """
        results = self.predict_sign(image)
        bboxes = self.return_bboxes(results)
        signs = self.crop_signs(image, bboxes)
        self.args_handler(image, bboxes, signs)
        return signs

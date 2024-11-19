from ultralytics import YOLO
import cv2

class SignRecognition(YOLO):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, task="detect")
        self.show_images = kwargs.get('show_images', False)
        self.show_signs = kwargs.get('show_signs', False)
        self.prob_threshold = kwargs.get('prob_threshold', 0.4)
        self.iou_threshold = kwargs.get('iou_threshold', 0.3)
        self.model_imgsz = kwargs.get('model_imgsz', 640)

    def predict_sign(self, data):
        return self.predict(source=data, conf=self.prob_threshold, device="cpu", imgsz=self.model_imgsz)[0]

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
        cv2.imshow('Current Frame', image_)

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
    def check_iou(box1, box2):
        x1, y1, x2, y2, _, _ = box1
        x1_, y1_, x2_, y2_, _, _ = box2
        xA = max(x1, x1_)
        yA = max(y1, y1_)
        xB = min(x2, x2_)
        yB = min(y2, y2_)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2Area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def return_unique_bboxes(self, results):
        boxes = []
        for box in results.boxes.data.tolist():
            if not boxes:
                boxes.append(box)
            else:
                for box_ in boxes:
                    if self.check_iou(box, box_) < self.iou_threshold:
                        boxes.append(box)
                    elif box[4] > box_[4]:
                        boxes.remove(box_)
                        boxes.append(box)
        return boxes

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
        bboxes = self.return_unique_bboxes(output)
        signs, results = self.crop_signs(image, bboxes)
        self.args_handler(image, bboxes, signs, results)
        return signs, results

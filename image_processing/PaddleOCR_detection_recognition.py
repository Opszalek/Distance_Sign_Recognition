from paddleocr import PaddleOCR, draw_ocr
import cv2

class PaddleOCR_sign:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # font path for drawing text on image
        self.font_path = '../Models/DejaVuSans-Bold.ttf'

    def predict_text(self, image):
        result = self.ocr.ocr(image, cls=True)
        return result[0]

    def predict_and_draw(self, image):
        result = self.predict_text(image)
        print(f"Result for : {result}")
        img = image
        if result is None or len(result) == 0 or result[0] is None:
            print(f"No text detected in image: {image}")
            return None
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]
        img_annotated = draw_ocr(img, boxes, txts, scores, font_path=self.font_path)
        cv2.imshow('image', img_annotated)
        cv2.waitKey(0)
        return img_annotated

    def predict_and_print(self, image):
        result = self.predict_text(image)
        if result is None or len(result) == 0 or result[0] is None:
            print(f"No text detected in image: {image}")
            return None
        print(f"Result for {image}: {result}")
        return result
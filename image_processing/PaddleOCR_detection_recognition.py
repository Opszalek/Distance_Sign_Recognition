from paddleocr import PaddleOCR

class PaddleOCR_sign:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.font_path = '../Models/DejaVuSans-Bold.ttf'

    def predict_text(self, image):
        result = self.ocr.ocr(image, cls=True)
        return result[0]
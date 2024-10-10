from paddleocr import PaddleOCR

class PaddleOCR_sign:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, det_db_thresh = 0.4, det_db_box_thresh = 0.5,det_db_unclip_ratio = 1.4, max_batch_size = 32,
                det_limit_side_len = 1000, det_db_score_mode = "slow", dilation = False, lang='en', ocr_version = "PP-OCRv4")
        self.font_path = '../Models/DejaVuSans-Bold.ttf'

    def predict_text(self, image):
        result = self.ocr.ocr(image, cls=True)
        return result[0]
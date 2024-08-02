import easyocr
import cv2

class EasyOCR_sign:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def predict_text(self, images):
        texts = []
        for image in images:
            result = self.reader.readtext(image, allowlist='0123456789')
            texts.append(result)
        return texts

    def predict_and_draw(self, image):
        result = self.predict_text(image)
        img = cv2.imread(image)
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        return img

    def predict_and_print(self, image):
        result = self.predict_text(image)
        print(f"Result for {image}: {result}")
        return result
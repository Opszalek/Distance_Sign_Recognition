import easyocr
import os
import cv2


class TextRecognition:
    def __init__(self, path_to_save_cropped='../TESTS/cropped'):
        self.reader = easyocr.Reader(['en'], recog_network='ocr_trainer', detector=False,
                                     user_network_directory='../Models/Text_recognition/ocr_trainer',
                                     model_storage_directory='../Models/Text_recognition/ocr_trainer')
        self.debug = False

    def predict_text(self, cropped_text):
        if self.debug:
            return self.reader.recognize(cropped_text, detail=1)
        else:
            return self.reader.recognize(cropped_text, detail=0)

    def predict_handler(self, images, debug=False):
        self.debug = debug
        results = []
        for image in images:
            result = self.predict_text(image)
            results.append(result)
        return results


if __name__ == "__main__":
    # main for debuging reasons
    tr = TextRecognition()
    for file in os.listdir('../Dataset/Examples/'):
        image = cv2.imread('../Dataset/Examples/' + file)
        result = tr.predict_handler(image)
        print(result, file)

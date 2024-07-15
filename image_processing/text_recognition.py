import easyocr
import os
import cv2

class TextRecognition:
    def __init__(self, path_to_save_cropped='../TESTS/cropped'):
        self.reader = easyocr.Reader(['en'], recog_network='ocr_trainer',detector=False,
                                     user_network_directory='../Models/Text_recognition/ocr_trainer',
                                     model_storage_directory='../Models/Text_recognition/ocr_trainer')
    def predict_text(self, image, debug=False):
        if debug:
            return self.reader.recognize(image, detail=1)
        else:
            return self.reader.recognize(image, detail=0)

if __name__ == "__main__":
    # main for debuging reasons
    tr = TextRecognition()
    for file in os.listdir('../Dataset/Examples/'):
        image = cv2.imread('../Dataset/Examples/'+ file)
        result = tr.predict_text(image)
        print(result, file)

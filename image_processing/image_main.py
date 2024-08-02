from utils.utils import timeit
from image_processing.sign_recognition import SignRecognition
import cv2
from image_processing.text_detection import TextDetection
from image_processing.text_recognition import TextRecognition
from image_processing.PaddleOCR_detection_recognition import PaddleOCR_sign
from image_processing.EasyOCR_detection_recognition import EasyOCR_sign
from image_processing import sign_tracker
import os


class SignTextRecognitionSystem:
    def __init__(self, path_to_model,
                 show_images=True, save_cropped=True, ocr='paddle', **kwargs):
        self.results_path = kwargs.get('results_path', '../Dataset/output')
        self.crops_path = kwargs.get('crops_path', '../Dataset/crops')
        self.frames_path = kwargs.get('frames_path', '../Dataset/frame')

        self.save_results_ = kwargs.get('save_results', False)
        self.show_images = show_images
        self.save_cropped = save_cropped

        self.sign_recognition = SignRecognition(
            path_to_model, path_to_save_cropped=self.crops_path, show_images=show_images, save_cropped=save_cropped)
        self.tracker = sign_tracker.SignTracker()
        self.text_rec = TextDetection()
        self.text_det = TextRecognition()
        self.text_det_rec_paddle = PaddleOCR_sign()
        self.text_det_rec_easy = EasyOCR_sign()
        self.ocr = self.return_ocr(ocr_type=ocr)
        self.cropped_sign_number = self.return_last_number()

    def detect_signs(self, image):
        return self.sign_recognition.process_image(image)

    def detect_text_Paddle(self, sign):
        return self.text_det_rec_paddle.predict_text(sign)

    def detect_text_Easy(self, sign):
        return self.text_det_rec_easy.predict_text(sign)

    def track_signs(self, signs, results, image=None):
        return self.tracker.handle_tracking(list(zip(signs, results)))

    def annotate_sign(self, sign, text_data):
        sign_ = sign.copy()
        if text_data and text_data[0] is not None:
            for text_info in text_data[0]:
                if text_info:
                    box, (detected_text, confidence) = text_info
                    for i in range(len(box)):
                        cv2.line(sign_, tuple(map(int, box[i])), tuple(map(int, box[(i + 1) % len(box)])),
                                 (0, 255, 0),
                                 2)
                    cv2.putText(sign_, f"{detected_text}:{confidence:.3f}",
                                (int(box[0][0]), int(box[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return sign_

    def display_sign_text(self, signs, texts):
        for sign, text_data in zip(signs, texts):
            sign = self.annotate_sign(sign, text_data)
            cv2.imshow('Sign', sign)

    def return_ocr(self, ocr_type=None):
        # OCR should have predict_text method which takes list of images and returns [bbox, (text, confidence)]
        if ocr_type == 'paddle':
            return self.text_det_rec_paddle
        elif ocr_type == 'easy':
            return self.text_det_rec_easy
        else:
            raise ValueError("Invalid OCR type. Choose 'paddle' or 'easy'.")

    def return_last_number(self):
        # TODO: add reading last number from file
        os.makedirs(os.path.join(self.results_path, 'labels'),
                    exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'images_annotated'),
                    exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'images'),
                    exist_ok=True)

        return 1

    def save_results(self, signs, texts):
        text_to_save = []
        for sign, text_data in zip(signs, texts):
            annotated_sign = self.annotate_sign(sign, text_data)
            if text_data[0] is not None:
                for text_info in text_data[0]:
                    box, (detected_text, confidence) = text_info
                    text_to_save.append([box, detected_text, confidence])

            with open(self.results_path + f'/labels/results_{self.cropped_sign_number}.txt', 'w') as f:
                f.write(f"sign_{self.cropped_sign_number}.png\n")
                for line in text_to_save:
                    f.write(f"{line}\n")

            cv2.imwrite(self.results_path +
                        f'/images_annotated/sign_annotated_{self.cropped_sign_number}.png', annotated_sign)
            cv2.imwrite(self.results_path + f'/images/sign_{self.cropped_sign_number}.png', sign)
            self.cropped_sign_number += 1

    def args_handler(self, image, signs, texts):
        if self.save_results_:
            self.save_results(signs, texts)

    def process_image(self, image):
        signs, results = self.detect_signs(image)
        selected_signs, selected_results = self.track_signs(signs, results)
        self.tracker.draw_bboxes(image)
        if len(selected_signs) == 0:
            return
        text_ = self.ocr.predict_text(selected_signs)
        self.display_sign_text(selected_signs, text_)
        #TODO: fix that args_handler works everytime instead only when len(selected_signs) > 0
        self.save_results_ = True
        self.args_handler(image, selected_signs, text_)


def get_images_from_directory(directory_path): # TODO: check if it works
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            yield cv2.imread(os.path.join(directory_path, filename))


def get_images_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def __main__():
    path_to_model = '../Models/Sign_recognition/model_1.pt'
    path_to_save_cropped = '../TESTS/cropped'
    image_source_type = 'video'  # Change to 'directory' if using directory of images
    image_source_path = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/Videos/dzien_video3.mp4'
    # image_source_path = '/path/to/your/image/directory'

    sign_text_recognition_system = SignTextRecognitionSystem(path_to_model)

    if image_source_type == 'video':
        image_generator = get_images_from_video(image_source_path)
    elif image_source_type == 'directory':
        image_generator = get_images_from_directory(image_source_path)
    else:
        raise ValueError("Invalid image source type. Choose 'video' or 'directory'.")

    for image in image_generator:
        sign_text_recognition_system.process_image(image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    __main__()

from utils.utils import timeit
import cv2
# from image_processing.text_detection import TextDetection
# from image_processing.text_recognition import TextRecognition
from image_processing.sign_recognition import SignRecognition
from image_processing.sign_segmentation import SignSegmentation
from image_processing.PaddleOCR_detection_recognition import PaddleOCR_sign
from image_processing.EasyOCR_detection_recognition import EasyOCR_sign
from image_processing import sign_tracker
import os
from datetime import datetime
import numpy as np


class SignTextRecognitionSystem:
    def __init__(self, **kwargs):
        self.results_path = kwargs.get('results_path', '../Dataset/output')
        self.frames_path = kwargs.get('frames_path', '../Dataset/frame')
        self.models_path = kwargs.get('models_path', '../Models')

        self.save_results = kwargs.get('save_results', False)
        self.save_frames = kwargs.get('save_frames', False)
        self.show_signs = kwargs.get('show_signs', False)
        self.show_segmentation_masks = kwargs.get('show_segmentation_masks', False)
        self.segmentation_type = kwargs.get('segmentation_type', 'yolov9c-seg')
        self.model_type = kwargs.get('model_type', 'yolov8')
        self.ocr_type = kwargs.get('ocr', 'paddle')
        self.show_images = kwargs.get('show_images', False)
        self.dst_std = 50
        self.std_hysteresis = 10
        self.bbox_height_threshold = 0.2

        self.date_hour = datetime.now().strftime("%d-%m-%Y_%H:%M")
        self.create_out_dir()
        self.cropped_sign_number = 1
        self.frame_number = 1

        self.sign_detection = self.return_detection_model(model_type=self.model_type)
        self.sign_segmentation = self.return_segmentation_model(model_type=self.segmentation_type)
        self.tracker = sign_tracker.SignTracker()
        # self.text_rec = TextDetection()
        # self.text_det = TextRecognition()
        self.text_det_rec_paddle = PaddleOCR_sign()
        self.text_det_rec_easy = EasyOCR_sign()
        self.ocr = self.return_ocr(ocr_type=self.ocr_type)

    def return_ocr(self, ocr_type=None):
        # OCR should have predict_text method which takes list of images and returns [[[bbox, (text, confidence)],[bbox, (text, confidence)]]]
        if ocr_type == 'paddle':
            return self.text_det_rec_paddle
        elif ocr_type == 'easy':
            return self.text_det_rec_easy
        else:
            raise ValueError("Invalid OCR type. Choose 'paddle' or 'easy'.")

    def return_detection_model(self, model_type=None):
        # Here you can add more models for sign recognition
        if model_type == 'yolov8':
            path_to_model = ('Sign_recognition/yolov8n_epochs_30_batch_16_dropout_0.1/content/runs/detect/train3'
                             '/weights/best.pt')
        elif model_type == 'yolov10':
            path_to_model = '/Sign_recognition/yolov10m_tiny_epochs_30_batch_16_dropout_0.1.pt'
        elif model_type == 'yolov8n':
            path_to_model = '/Sign_recognition/v8n_v2.pt'
        elif model_type == 'yolov9s':
            path_to_model = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Models/Test_dic/Sign_recognition/yolov9s_epochs_30_batch_16_dropout_0.1 (1)/content/runs/detect/train2/weights/best.pt'
        else:
            return 1

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignRecognition(path_to_model, show_images=self.show_images)

    def return_segmentation_model(self, model_type):
        # Here you can add more models for sign segmentation
        if model_type == 'yolov9c-seg':
            path_to_model = 'Sign_segmentation/yolov9c-seg_epochs_30_batch_16_dropout_0.1_daw.pt'
        elif model_type == 'yolov9c-seg-extended':
            path_to_model = '/Sign_segmentation/yolov9c-seg_epochs_30_batch_16_dropout_0.1_marc.pt'
        else:
            return 1

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignSegmentation(path_to_model, show_masks=self.show_segmentation_masks)

    def detect_signs(self, image):
        return self.sign_detection.process_image(image)

    def track_signs(self, signs, results, image=None):
        return self.tracker.handle_tracking(list(zip(signs, results)))

    def save_frame(self, image):
        cv2.imwrite(self.results_path + f'/frames/frame_{self.frame_number}.png', image)
        self.frame_number += 1

    def auto_contrast(self, image):
        contrast = np.std(image)
        # TODO: Fix when the brightness is too high
        if self.dst_std + self.std_hysteresis > contrast > self.dst_std:
            return image
        alpha = self.dst_std / contrast
        beta = 0
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    @staticmethod
    def annotate_sign(sign, text_data):
        sign_ = sign.copy()
        if text_data is not None:
            for text_info in text_data:
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

    def check_text_bboxes(self, texts, sign_shape):
        if texts:
            for text in texts:
                box, _ = text
                if box[3][1] - box[1][1] > sign_shape[0] * self.bbox_height_threshold:
                    return False
        return True

    def handle_text_detection(self, signs):
        text_data = []
        new_signs = []
        for sign in signs:
            adjusted_sign = self.auto_contrast(sign)
            texts = self.ocr.predict_text(adjusted_sign)
            if texts is None or not self.check_text_bboxes(texts, sign.shape):
                straight_sign = self.sign_segmentation.return_straight_sign(sign)
                adjusted_straight_sign = self.auto_contrast(straight_sign)
                texts = self.ocr.predict_text(adjusted_straight_sign)
                new_signs.append(adjusted_straight_sign)
            else:
                new_signs.append(adjusted_sign)
            text_data.append(texts)

        return text_data, new_signs

    def display_sign_text(self, signs, texts):
        for sign, text_data in zip(signs, texts):
            sign = self.annotate_sign(sign, text_data)
            cv2.imshow('Sign', sign)

    def create_out_dir(self):
        if self.save_results or self.save_frames:
            os.makedirs(os.path.join(self.results_path, self.date_hour), exist_ok=True)

            self.results_path = os.path.join(self.results_path, self.date_hour)

        if self.save_results:
            os.makedirs(os.path.join(self.results_path, 'labels'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'signs_annotated'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'signs'),
                        exist_ok=True)

        if self.save_frames:
            os.makedirs(os.path.join(self.results_path, 'frames'),
                        exist_ok=True)

    def save_result(self, signs, texts):
        text_to_save = []
        for sign, text_data in zip(signs, texts):
            annotated_sign = self.annotate_sign(sign, text_data)
            if text_data is not None:
                for text_info in text_data:
                    box, (detected_text, confidence) = text_info
                    text_to_save.append([box, detected_text, confidence])

            with open(self.results_path + f'/labels/results_{self.cropped_sign_number}.txt', 'w') as f:
                f.write(f"sign_{self.cropped_sign_number}.png\n")
                for line in text_to_save:
                    f.write(f"{line}\n")

            cv2.imwrite(self.results_path +
                        f'/signs_annotated/sign_annotated_{self.cropped_sign_number}.png', annotated_sign)
            cv2.imwrite(self.results_path + f'/signs/sign_{self.cropped_sign_number}.png', sign)
            self.cropped_sign_number += 1

    def args_handler(self, image, signs, texts):
        if self.save_results and len(signs) > 0:
            self.save_result(signs, texts)
        if self.save_frames:
            self.save_frame(image)
            cv2.imwrite(self.frames_path + f'/frame_{self.cropped_sign_number}.png', image)
        if self.show_signs:
            self.display_sign_text(signs, texts)

    def process_frame(self, image):
        signs, results = self.detect_signs(image)
        selected_signs, selected_results = self.track_signs(signs, results)
        self.tracker.draw_bboxes(image)
        text_, signs = self.handle_text_detection(selected_signs)
        self.args_handler(image, selected_signs, text_)

    def process_image(self, image):
        signs, results = self.detect_signs(image)
        # signs, selected_results = self.track_signs(signs, results)
        # self.tracker.draw_bboxes(image)'
        text_, signs = self.handle_text_detection(signs)
        self.args_handler(image, signs, text_)


def get_images_from_directory(directory_path):
    sorted_files = sorted(os.listdir(directory_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for filename in sorted_files:
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
    frame_number = 1
    image_source_type = 'video'  # choose 'video' or 'directory' video - for video, directory - for images
    video_source_path = '/home/opszalek/sign_cropped/Sign_cropped/s15_1_cropped.mp4'
    video_source_path = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/Videos/dzien_video3.mp4'
    # video_source_path = '/media/opszalek/C074672F7467277E/Users/Dawid/Videos/WonderFox Soft/HD Video Converter Factory Pro/OutputVideo/s11_2.mp4'
    image_source_path = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/output/05-08-2024_10:08!!!!!/images'

    sign_text_recognition_system = SignTextRecognitionSystem(model_type='yolov9s',
                                                             save_results=False, show_signs=True,
                                                             show_images=True, save_frames=False,
                                                             ocr='paddle')

    if image_source_type == 'video':
        image_generator = get_images_from_video(video_source_path)
        for image in image_generator:
            if frame_number > 2:
                sign_text_recognition_system.process_frame(image)
                cv2.waitKey(0)
            frame_number += 1
    elif image_source_type == 'directory':
        image_generator = get_images_from_directory(image_source_path)
        for image in image_generator:
            sign_text_recognition_system.process_image(image)
            cv2.waitKey(0)
    else:
        raise ValueError("Invalid image source type. Choose 'video' or 'directory'.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    __main__()

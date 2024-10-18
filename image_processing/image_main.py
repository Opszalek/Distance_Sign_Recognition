import cv2
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

        #Params when running the script
        self.save_results = kwargs.get('save_results', False)
        self.save_frames = kwargs.get('save_frames', False)
        self.show_signs = kwargs.get('show_signs', False)
        self.show_images = kwargs.get('show_images', False)
        self.show_segmentation_masks = kwargs.get('show_segmentation_masks', False)

        #Params for ROS2
        self.enable_preview = kwargs.get('enable_preview', False)

        self.segmentation_type = kwargs.get('segmentation_type', 'yolov8l-seg_cropped')
        self.model_type = kwargs.get('model_type', 'yolov8')
        self.ocr_type = kwargs.get('ocr', 'paddle')
        self.detection_type = kwargs.get('detection_type', 'normal')
        self.dst_std = 50
        self.std_hysteresis = 10
        self.bbox_height_threshold = 0.2

        self.date_hour = datetime.now().strftime("%d-%m-%Y_%H:%M")
        self.create_out_dir()
        self.cropped_sign_number = 1
        self.frame_number = 1

        self.sign_detection = self.return_detection_model(model_type=self.model_type)
        self.sign_segmentation = self.return_segmentation_model(model_type=self.segmentation_type)
        self.tracker = sign_tracker.SignTracker(enable_preview=self.enable_preview)
        self.text_det_rec_paddle = PaddleOCR_sign()
        self.text_det_rec_easy = EasyOCR_sign()
        self.ocr = self.return_ocr(ocr_type=self.ocr_type)

    def reset_system(self):
        self.cropped_sign_number = 1
        self.frame_number = 1
        self.create_out_dir()
        self.tracker = sign_tracker.SignTracker(enable_preview=self.enable_preview)

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
        if model_type == 'yolov8n':
            path_to_model = 'Sign_recognition/yolo8n.pt'
        elif model_type == 'yolov9t':
            path_to_model = 'Sign_recognition/yolov9t.pt'
        elif model_type == 'yolov9s':
            path_to_model = 'Sign_recognition/yolov9s.pt'
        elif model_type == 'yolov10n':
            path_to_model = 'Sign_recognition/yolov10n.pt'
        elif model_type == 'yolov10l':
            path_to_model = 'Sign_recognition/yolov10l.pt'
        else:
            return 1

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignRecognition(path_to_model, show_images=self.show_images)

    def return_segmentation_model(self, model_type):
        # Here you can add more models for sign segmentation
        if model_type == 'yolov9c-seg':
            path_to_model = 'Sign_segmentation/yolov9c-seg_epochs_30_batch_16_dropout_0.1_daw.pt'
        elif model_type == 'yolov9c-seg-extended':
            path_to_model = 'Sign_segmentation/yolov9c-seg-extended.pt'
        elif model_type == 'yolov8l-seg-cropped':
            path_to_model = 'Sign_segmentation/yolov8l-seg-cropped.pt'
        else:
            return 1

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignSegmentation(path_to_model, show_masks=self.show_segmentation_masks)

    def detect_signs(self, image):
        return self.sign_detection.process_image(image)

    def track_signs(self, signs, results, image):
        return self.tracker.handle_tracking(list(zip(signs, results)), image)

    def save_frame(self, image):
        cv2.imwrite(self.results_path + f'/frames/frame_{self.frame_number}.png', image)
        self.frame_number += 1

    def auto_contrast(self, image):
        contrast = np.std(image)
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
                    cv2.putText(sign_, f"{detected_text}",#:{confidence:.3f}",
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

    @staticmethod
    def add_background(image):
        h, w, _ = image.shape
        max_size = max(h, w)
        background = cv2.copyMakeBorder(image, 0, max_size - h, 0, max_size - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return cv2.resize(background, (640, 640))

    def detect_text_dual(self, signs):
        text_data = []
        adjusted_signs = []
        for sign in signs:
            adjusted_sign = self.auto_contrast(sign)
            texts = self.ocr.predict_text(adjusted_sign)
            if texts is None or not self.check_text_bboxes(texts, sign.shape):
                straight_sign = self.sign_segmentation.return_straight_sign(sign)
                adjusted_straight_sign = self.auto_contrast(straight_sign)
                texts = self.ocr.predict_text(adjusted_straight_sign)
                adjusted_signs.append(adjusted_straight_sign)
            else:
                adjusted_signs.append(adjusted_sign)
            text_data.append(texts)

        return text_data, adjusted_signs

    def detect_text_dual_extended(self, signs):
        text_data = []
        adjusted_signs = []
        for sign in signs:
            adjusted_sign = self.auto_contrast(sign)
            texts = self.ocr.predict_text(adjusted_sign)
            if texts is None or not self.check_text_bboxes(texts, sign.shape):
                straight_sign = self.sign_segmentation.return_straight_sign(sign)
                adjusted_straight_sign = self.auto_contrast(straight_sign)
                straight_background = self.add_background(adjusted_straight_sign)
                texts = self.ocr.predict_text(straight_background)
                adjusted_signs.append(adjusted_straight_sign)
            else:
                adjusted_signs.append(adjusted_sign)
            text_data.append(texts)

        return text_data, adjusted_signs

    def detect_text_contrast_straight(self, signs):
        text_data = []
        adjusted_signs = []
        for sign in signs:
            straight_sign = self.sign_segmentation.return_straight_sign(sign)
            adjusted_straight_sign = self.auto_contrast(straight_sign)
            texts = self.ocr.predict_text(adjusted_straight_sign)
            adjusted_signs.append(adjusted_straight_sign)
            text_data.append(texts)

        return text_data, adjusted_signs

    def detect_text_straight_extended(self, signs):
        text_data = []
        adjusted_signs = []
        for sign in signs:
            straight_sign = self.sign_segmentation.return_straight_sign(sign)
            adjusted_straight_sign = self.auto_contrast(straight_sign)
            straight_background = self.add_background(adjusted_straight_sign)
            texts = self.ocr.predict_text(straight_background)
            adjusted_signs.append(adjusted_straight_sign)
            text_data.append(texts)

        return text_data, adjusted_signs

    def detect_text_contrast(self, signs):
        text_data = []
        adjusted_signs = []
        for sign in signs:
            adjusted_sign = self.auto_contrast(sign)
            texts = self.ocr.predict_text(adjusted_sign)
            adjusted_signs.append(adjusted_sign)
            text_data.append(texts)

        return text_data, adjusted_signs

    def detect_text(self, signs):
        text_data = []
        for sign in signs:
            texts = self.ocr.predict_text(sign)
            text_data.append(texts)

        return text_data, signs

    def handle_text_detection(self, signs):
        if self.detection_type == 'contrast':
            return self.detect_text_contrast(signs)
        elif self.detection_type == 'contrast_straighten_normal':
            return self.detect_text_dual(signs)
        elif self.detection_type == 'contrast_straighten_normal_background':
            return self.detect_text_dual_extended(signs)
        elif self.detection_type == 'normal':
            return self.detect_text(signs)
        elif self.detection_type == 'contrast_straighten':
            return self.detect_text_contrast_straight(signs)
        elif self.detection_type == 'contrast_straighten_background':
            return self.detect_text_straight_extended(signs)


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
        if self.show_signs:
            self.display_sign_text(signs, texts)

    def process_frame(self, image):
        """
        Process frame, follow sign crop it and detect text
        :param image: image to process
        :return: list of cropped signs and text
        """
        signs, results = self.detect_signs(image)
        selected_frames, selected_signs, selected_results, annotated_image = self.track_signs(signs, results, image)
        text, signs_adjusted = self.handle_text_detection(selected_signs)
        self.args_handler(image, signs_adjusted, text)
        if self.enable_preview:
            return signs_adjusted, text, annotated_image
        return signs_adjusted, text

    def process_image(self, image):
        """
        Process image and return cropped signs and text
        """
        signs, results = self.detect_signs(image)
        text, signs_adjusted = self.handle_text_detection(signs)
        self.args_handler(image, signs_adjusted, text)
        return signs_adjusted, text

    def frame_selector(self, image):
        """
        Process image and return cropped signs and whole frame
        """
        signs, results = self.detect_signs(image)
        selected_frames, selected_signs, selected_results, annotated_image = self.track_signs(signs, results, image)
        return selected_signs, selected_results, annotated_image

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
    image_source_type = 'video'  # choose 'video' or 'directory' video - for video, directory - for images
    video_source_path = '/home/opszalek/ALL_PIKIETAZ_VIDEOS/TEST_MP4.mp4'#'/path/to/video.mp4'
    image_source_path = '/home/opszalek/Projekt_pikietaz/Distance_Sign_Recognition/Dataset/output/15-08-2024_21:45/frames'#'/path/to/directory/with/images'

    sign_text_recognition_system = SignTextRecognitionSystem(model_type='yolov10n', segmentation_type='yolov8l-seg-cropped',
                                                             save_results=False, show_signs=True,
                                                             show_images=True, save_frames=False,
                                                             enable_preview=False,
                                                             ocr='paddle', detection_type='contrast_straighten')

    if image_source_type == 'video':
        image_generator = get_images_from_video(video_source_path)
        for image in image_generator:
            signs_adjusted, text, annotated_image = sign_text_recognition_system.process_frame(image)
            print(text)
            annotated_image = cv2.resize(annotated_image, (640, 640))
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
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

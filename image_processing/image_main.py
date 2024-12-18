import cv2
from image_processing.sign_recognition import SignRecognition
from image_processing.sign_segmentation import SignSegmentation
from image_processing.PaddleOCR_detection_recognition import PaddleOCR_sign
# from image_processing.EasyOCR_detection_recognition import EasyOCR_sign
from image_processing import sign_tracker
import os
from datetime import datetime
import numpy as np


class SignTextRecognitionSystem:
    def __init__(self, **kwargs):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
        self.results_path = kwargs.get('results_path', os.path.join(base_dir, 'output'))
        self.frames_path = kwargs.get('frames_path', os.path.join(base_dir, 'frame'))
        self.models_path = kwargs.get('models_path', os.path.abspath('../Models'))
        self.system_version = kwargs.get('system_version', 'Linux')
        self.error_callback = kwargs.get('error_callback', None)

        #Params when running the script
        self.save_results = kwargs.get('save_results', False)
        self.save_signs = kwargs.get('save_signs', False)
        self.show_signs = kwargs.get('show_signs', False)
        self.show_images = kwargs.get('show_images', False)
        self.show_segmentation_masks = kwargs.get('show_segmentation_masks', False)

        #Params for ROS2
        self.enable_preview = kwargs.get('enable_preview', False)

        self.device_type = kwargs.get('device_type', '')
        self.segmentation_type = kwargs.get('segmentation_type', 'yolov8l-seg_cropped')
        self.model_type = kwargs.get('model_type', 'yolov8')
        self.ocr_type = kwargs.get('ocr', 'paddle')
        self.detection_type = kwargs.get('detection_type', 'normal')
        self.dst_std = 50
        self.std_hysteresis = 10
        self.bbox_height_threshold = 0.2

        self.date_hour = datetime.now().strftime("%d-%m-%Y_%H-%M")
        self.create_out_dir()
        self.cropped_sign_number = 1
        self.frame_number = 1
        self.last_timestamp = 0

        self.sign_detection = self.return_detection_model(model_type=self.model_type)
        self.sign_segmentation = self.return_segmentation_model(model_type=self.segmentation_type)
        self.tracker = sign_tracker.SignTracker(enable_preview=self.enable_preview)
        self.text_det_rec_paddle = PaddleOCR_sign()
        # self.text_det_rec_easy = EasyOCR_sign()
        self.ocr = self.return_ocr(ocr_type=self.ocr_type)

    def error_callback_(self, error_type):
        if error_type == 'model':
            error = "Invalid model type. Check available models in return_detection_model and return_segmentation_model methods."
        elif error_type == 'text_detection':
            error = "Invalid text detection type. Check available types in handle_text_detection method."
        elif error_type == 'ocr_type':
            error = "Invalid OCR type. Check available types in return_ocr method."
        else:
            error = "Error occurred. Check error_callback_ method."

        if self.error_callback:
            self.error_callback(error)
        else:
            raise ValueError(error)

    def reset_system(self):
        self.cropped_sign_number = 1
        self.frame_number = 1
        self.create_out_dir()
        self.tracker = sign_tracker.SignTracker(enable_preview=self.enable_preview)

    def return_ocr(self, ocr_type=None):
        # OCR should have predict_text method which takes list of images and returns [[[bbox, (text, confidence)],[bbox, (text, confidence)]]]
        if ocr_type == 'paddle':
            return self.text_det_rec_paddle
        # elif ocr_type == 'easy':
        #     return self.text_det_rec_easy
        else:
            self.error_callback_('ocr_type')

    def return_detection_model(self, model_type=None):
        # Here you can add more models for sign recognition
        if self.system_version == 'Linux':
            if model_type == 'yolov8n':
                path_to_model = 'Sign_recognition/yolov8n.pt'
                model_image_size = 640
            elif model_type == 'yolov8n_cpu':
                path_to_model = 'Sign_recognition/yolov8n_int8_openvino_model/'
                model_image_size = 640
            elif model_type == 'yolov8n_cpu_480':
                path_to_model = 'Sign_recognition/best_int8_openvino_model_480/'
                model_image_size = 480
            else:
                self.error_callback_('model')

        elif self.system_version == 'Windows':
            if model_type == 'yolov8n':
                path_to_model = r'Sign_recognition\yolov8n.pt'
                model_image_size = 640
            elif model_type == 'yolov8n_cpu':
                path_to_model = r'Sign_recognition\yolov8n_int8_openvino_model/'
                model_image_size = 640
            elif model_type == 'yolov8n_cpu_480':
                path_to_model = r'Sign_recognition\best_int8_openvino_model_480/'
                model_image_size = 480
            else:
                self.error_callback_('model')

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignRecognition(path_to_model, show_images=self.show_images, model_imgsz=model_image_size, device=self.device_type)

    def return_segmentation_model(self, model_type):
        # Here you can add more models for sign segmentation
        if self.system_version == 'Linux':
            if model_type == 'yolov9c-seg-extended':
                path_to_model = 'Sign_segmentation/yolov9c-seg-extended.pt'
            elif model_type == 'yolov8l-seg-cropped':
                path_to_model = 'Sign_segmentation/yolov8l-seg-cropped.pt'
            elif model_type == 'yolov8l-seg-cropped-cpu':
                path_to_model = 'Sign_segmentation/yolov8l-seg-cropped_int8_openvino_model/'
            else:
                self.error_callback_('model')

        elif self.system_version == 'Windows':
            if model_type == 'yolov9c-seg-extended':
                path_to_model = r'Sign_segmentation\yolov9c-seg-extended.pt'
            elif model_type == 'yolov8l-seg-cropped':
                path_to_model = r'Sign_segmentation\yolov8l-seg-cropped.pt'
            elif model_type == 'yolov8l-seg-cropped-cpu':
                path_to_model = r'Sign_segmentation\yolov8l-seg-cropped_int8_openvino_model/'
            else:
                self.error_callback_('model')

        path_to_model = os.path.join(self.models_path, path_to_model)
        return SignSegmentation(path_to_model, show_masks=self.show_segmentation_masks, device=self.device_type)

    def detect_signs(self, image):
        return self.sign_detection.process_image(image)

    def track_signs(self, signs, results, image):
        return self.tracker.handle_tracking(list(zip(signs, results)), image)

    def save_frame(self, image, timestamp):
        frame_path = os.path.join(self.results_path, 'frames', f'{timestamp}.png')
        cv2.imwrite(frame_path, image)
        self.frame_number += 1

    def save_sign(self, sign, timestamp):
        sign_path = os.path.join(self.results_path, 'signs', f'{timestamp}.png')
        cv2.imwrite(sign_path, sign)

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
        else:
            self.error_callback_('text_detection')


    def display_sign_text(self, signs, texts):
        for sign, text_data in zip(signs, texts):
            sign = self.annotate_sign(sign, text_data)
            cv2.imshow('Sign', sign)

    def create_out_dir(self):
        if self.save_results or self.save_signs:
            os.makedirs(os.path.join(self.results_path, self.date_hour), exist_ok=True)

            self.results_path = os.path.join(self.results_path, self.date_hour)

        if self.save_results:
            os.makedirs(os.path.join(self.results_path, 'labels'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'signs_annotated'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'signs'),
                        exist_ok=True)

        if self.save_signs:
            os.makedirs(os.path.join(self.results_path, 'frames'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'signs'),
                        exist_ok=True)
            os.makedirs(os.path.join(self.results_path, 'text'),
                        exist_ok=True)

    def save_result(self, signs, texts):
        text_to_save = []
        for sign, text_data in zip(signs, texts):
            annotated_sign = self.annotate_sign(sign, text_data)
            if text_data is not None:
                for text_info in text_data:
                    box, (detected_text, confidence) = text_info
                    text_to_save.append([box, detected_text, confidence])

            labels_dir = os.path.join(self.results_path, 'labels')
            os.makedirs(labels_dir, exist_ok=True)

            results_file = os.path.join(labels_dir, f'results_{self.cropped_sign_number}.txt')
            with open(results_file, 'w') as f:
                f.write(f"sign_{self.cropped_sign_number}.png\n")
                for line in text_to_save:
                    f.write(f"{line}\n")

            annotated_dir = os.path.join(self.results_path, 'signs_annotated')
            os.makedirs(annotated_dir, exist_ok=True)
            annotated_file = os.path.join(annotated_dir, f'sign_annotated_{self.cropped_sign_number}.png')
            cv2.imwrite(annotated_file, annotated_sign)

            signs_dir = os.path.join(self.results_path, 'signs')
            os.makedirs(signs_dir, exist_ok=True)
            sign_file = os.path.join(signs_dir, f'sign_{self.cropped_sign_number}.png')
            cv2.imwrite(sign_file, sign)

            # Increment the sign counter
            self.cropped_sign_number += 1

    def args_handler(self, signs, sign_frames, texts=None, timestamp=None):
        if self.save_results and len(signs) > 0 and texts:
            self.save_result(signs, texts)
        if self.save_signs and len(signs) > 0:
            self.save_frame(sign_frames[0], timestamp)
            self.save_sign(signs[0], timestamp)
        if self.show_signs and texts:
            self.display_sign_text(signs, texts)

    def process_frame(self, image, timestamp):
        """
        Process frame, follow sign crop it and detect text
        :param image: image to process
        :param timestamp: timestamp of the image
        :return: list of cropped signs and text
        """
        signs, results = self.detect_signs(image)
        selected_frames, selected_signs, selected_results, annotated_image = self.track_signs(signs, results, image)
        text, signs_adjusted = self.handle_text_detection(selected_signs)
        self.args_handler(signs_adjusted, selected_frames, texts=text, timestamp=timestamp)
        if self.enable_preview:
            return signs_adjusted, text, annotated_image
        return signs_adjusted, text

    def process_image(self, image, timestamp):
        """
        Process image and return cropped signs and text
        """
        signs, results = self.detect_signs(image)
        text, signs_adjusted = self.handle_text_detection(signs)
        self.args_handler(image, signs_adjusted, text, timestamp=timestamp)
        return signs_adjusted, text

    def frame_selector(self, image, timestamp):
        """
        Process image and return cropped signs and whole frame
        """
        signs, results = self.detect_signs(image)
        selected_frames, selected_signs, selected_results, annotated_image = self.track_signs(signs, results, image)
        self.args_handler(selected_signs, selected_frames, timestamp=timestamp)
        if self.enable_preview:
            return selected_signs, selected_results, annotated_image
        else:
            return selected_signs, selected_frames

    def process_sign_images(self, continue_processing=False):
        """
        Looks for images in folder and processes them. Can be stopped and will continue from last image.
        """
        images_path = os.path.join(self.results_path, 'signs')
        test_result_path = os.path.join(self.results_path, 'text')
        image_iterator = get_images_from_directory(images_path, sort_type=1)
        for sign, timestamp in image_iterator:
            if timestamp > self.last_timestamp:
                text, signs_adjusted = self.handle_text_detection([sign])
                with open(os.path.join(self.results_path, 'text',f'{timestamp}.txt'), 'w') as f:
                    f.write(f"{text}\n")
                self.last_timestamp = timestamp
                if not continue_processing:
                    break

def get_images_from_directory(directory_path, sort_type=0):
    if sort_type == 1:
        sorted_files = sorted(os.listdir(directory_path), key=lambda x: int(x.split('.')[0]))
        for filename in sorted_files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                timestamp = int(filename.split('.')[0])
                yield cv2.imread(os.path.join(directory_path, filename)), timestamp
    else:
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

    sign_text_recognition_system = SignTextRecognitionSystem(model_type='yolov8n_cpu_480', segmentation_type='yolov8l-seg-cropped-cpu',
                                                             save_results=False, show_signs=True,
                                                             show_images=True, save_signs=True,
                                                             enable_preview=True,
                                                             ocr='paddle', detection_type='contrast_straighten',
                                                             device_type='cpu')

    # sign_text_recognition_system.last_timestamp=15
    # for i in range(10):
    #     sign_text_recognition_system.process_sign_images()
    if image_source_type == 'video':
        image_generator = get_images_from_video(video_source_path)
        for image in image_generator:
            signs_adjusted, text, annotated_image = sign_text_recognition_system.process_frame(image, timestamp=datetime.now())
            # selected_signs, selected_results, annotated_image = sign_text_recognition_system.frame_selector(image, timestamp=datetime.now())
            # print(text)
            annotated_image = cv2.resize(annotated_image, (640, 640))
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(1)
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

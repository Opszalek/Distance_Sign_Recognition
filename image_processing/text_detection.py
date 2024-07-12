import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

base_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..'))
mixnet_dir = os.path.join(project_root, 'utils', 'MixNet')


module_paths = [
    project_root,
    mixnet_dir
]

for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

from network.textnet import TextNet
from cfglib.config import config as cfg, update_config
from cfglib.option import BaseOptions
from util.visualize import visualize_detection


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end-start} seconds to run')
        return result
    return wrapper

class TextDetection:
    def __init__(self):
        # Parse arguments
        option = BaseOptions()
        args = option.initialize()
        update_config(cfg, args)

        self.model = TextNet(is_training=False, backbone=cfg.net)
        model_path = os.path.join("../Models/Text_detection", cfg.exp_name, f'MixNet_{cfg.net}_{cfg.checkepoch}.pth')
        self.model.load_model(model_path)
        self.model.to(cfg.device)
        self.model.eval()


    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_shape = image.shape[:2]
        image = cv2.resize(image, (cfg.test_size[1], cfg.test_size[0]))
        image = image.astype(np.float32) / 255.0
        image -= np.array(cfg.means)
        image /= np.array(cfg.stds)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(cfg.device)
        return image


    def save_cropped_regions(self, img, b_boxes, output_dir, file):
        # function to save cropped regions of the image for debugging purposes

        for i, bbox in enumerate(b_boxes):
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            x1 = int(x1 / cfg.test_size[1] * self.original_shape[1])
            y1 = int(y1 / cfg.test_size[0] * self.original_shape[0])
            x2 = int(x2 / cfg.test_size[1] * self.original_shape[1])
            y2 = int(y2 / cfg.test_size[0] * self.original_shape[0])


            cropped_text = img[y1:y2, x1:x2]

            file_name = file.split('.')[0]

            cropped_output_path = os.path.join(output_dir, f'cropped_{file_name}_{i}.png')
            cv2.imwrite(cropped_output_path, cropped_text)
            print(f"Saved cropped text to {cropped_output_path}")

    def return_cropped_regions(self, img, b_boxes):
        cropped_texts = []
        for i, bbox in enumerate(b_boxes):
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            x1 = int(x1 / cfg.test_size[1] * self.original_shape[1])
            y1 = int(y1 / cfg.test_size[0] * self.original_shape[0])
            x2 = int(x2 / cfg.test_size[1] * self.original_shape[1])
            y2 = int(y2 / cfg.test_size[0] * self.original_shape[0])

            cropped_text = img[y1:y2, x1:x2]
            cropped_texts.append(cropped_text)
        return cropped_texts

    def img_visualize(self, image, output_dict, original_image):
        img_show = image[0].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * np.array(cfg.stds) + np.array(cfg.means)) * 255).astype(np.uint8)
        detection_image, bounding_boxes = visualize_detection(img_show, output_dict)
        detection_image_resized = cv2.resize(detection_image, (original_image.shape[1], original_image.shape[0]))
        return detection_image_resized, bounding_boxes

    def save_img_visualize(self, detection_image, output_dir):
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, detection_image)
        print(f"Saved detection image to {output_path}")

    def detect_text(self, image):
        with torch.no_grad():
            preprocessed_image = self.preprocess_image(image)
            output_dict = self.model({'img': preprocessed_image})
            detection_image, bounding_boxes = self.img_visualize(preprocessed_image, output_dict, image)
            cropped_texts = self.return_cropped_regions(image, bounding_boxes)
            return cropped_texts

@timeit
def main(image_path, output_dir, file):
    cudnn.benchmark = False

    image = cv2.imread(image_path)

    original_image = image.copy()
    text_recognizer = TextDetection()
    with torch.no_grad():
        preprocessed_image = text_recognizer.preprocess_image(image)
        output_dict = text_recognizer.model({'img': preprocessed_image})
        detection_image, bounding_boxes = text_recognizer.img_visualize(preprocessed_image, output_dict, original_image)
        text_recognizer.save_img_visualize(detection_image, output_dir)
        text_recognizer.save_cropped_regions(original_image, bounding_boxes, output_dir, file)


if __name__ == "__main__":
    input_dir = os.path.join(project_root, "Dataset", "znaki")

    for file in os.listdir(input_dir):
        print(file)
        text_recognizer = TextDetection()
        image_path = os.path.join(input_dir, file)
        output_dir = 'output'
        cropped_texts = text_recognizer.detect_text(cv2.imread(image_path))
        for cropped_text in cropped_texts:
            cv2.imshow("cropped_texts", cropped_text)
            cv2.waitKey(0)

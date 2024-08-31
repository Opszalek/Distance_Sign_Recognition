from ultralytics import YOLO
import cv2
from image_processing import sign_tracker
import torch
import numpy as np

class SignSegmentation(YOLO):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path)
        self.show_masks = kwargs.get('show_masks', False)

    def segment_sign(self, data):
        return self(data)[0]

    def show_mask(self, image, contour):
        mask = cv2.drawContours(np.zeros(image.shape[:2], dtype=np.uint8),
                                [contour], -1, 255, thickness=cv2.FILLED)

        resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:, :, 0] = resized_mask * 255
        colored_mask[:, :, 1] = resized_mask * 150
        colored_mask[:, :, 2] = resized_mask * 150

        masked_image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

        cv2.imshow('Segmentation Mask', masked_image)

    def return_straight_sign(self, image):
        warped = image
        results = self(image)
        for result in results:
            for i, det in enumerate(result):
                contour = det.masks.xy.pop()
                contour = contour.astype(np.int32)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                if self.show_masks:
                    self.show_mask(image, contour)

                width = int(rect[1][0])
                height = int(rect[1][1])
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height - 1],
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                warped = cv2.warpPerspective(image, M, (width, height))

                if width > height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped
import cv2
import pywt

import numpy as np

class Sign:
    def __init__(self, new_sign, ID, image, use_sharpness=False):
        sign_img, (x1, y1, x2, y2, score, class_id) = new_sign
        self.bbox = (x1, y1, x2, y2)
        self.score = score
        self.class_id = class_id
        self.last_seen = 0
        self.ID = ID
        self.sharpness_func = Sign.canny_score
        self.sign_img = sign_img
        self.frame_img = image
        self.sharpness_score = self.sharpness_func(cv2.resize(self.sign_img, (500, 120)))
        self.last_bboxes = []
        self.last_bboxes.append(self.bbox)
        self.distance_delta = [0, 0]
        self.use_sharpness = use_sharpness

    def return_sign(self):
        return self.sign_img

    def return_frame(self):
        return self.frame_img

    def return_results(self):
        return self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.score, self.class_id

#DEFINIED SHARPNESS SCORE FUNCTIONS---------------------------------------------------
    @staticmethod
    def laplacian_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    @staticmethod
    def sobel_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        variance = sobel_magnitude.var()
        return variance

    @staticmethod
    def wavelet_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coeffs = pywt.wavedec2(gray, 'db1', level=2)
        details = coeffs[1:]
        score = 0
        for level in details:
            LH, HL, HH = level
            score += np.mean(np.abs(HH))
        return score

    @staticmethod
    def canny_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        score = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        return score
#--------------------------------------------------------------------------------------

    def compare_sharpness_handler(self, new_sign_img):
        new_sharpness_score = self.sharpness_func(new_sign_img)
        if new_sharpness_score >= self.sharpness_score:
            self.sharpness_score = new_sharpness_score
            self.sign_img = new_sign_img

    def increase_last_seen(self):
        self.last_seen += 1

    def update_sign(self, new_sign, distance_delta, image):
        new_sign_image, (x1, y1, x2, y2, score, class_id) = new_sign
        if self.use_sharpness:
            self.compare_sharpness_handler(new_sign_image)
        else:
            self.sign_img = new_sign_image
        self.distance_delta = distance_delta
        self.bbox = (x1, y1, x2, y2)
        self.last_bboxes.append(self.bbox)
        self.score = score
        self.last_seen = 0
        self.frame_img = image

    def get_bbox(self):
        return self.bbox


class SignTracker:
    def __init__(self, debug_mode=False, enable_preview=False):
        self.signs = {}
        self.ID = 1
        self.no_detection_threshold = 1
        self.IOU_threshold = 0.75 #0.55
        self.score_threshold = 0.4#0.55
        self.image_size = [2048, 2048]
        self.ROI = [1024, 1600, 150, 60]  #x1,y1,x_offset,y_offset
        self.last_bbox = None
        self.width_expansion_factor = 3.0
        self.height_expansion_factor = 1.35
        self.curr_image = None
        self.debug_mode = debug_mode
        self.enable_preview = enable_preview
        print(f'Enable preview: {self.enable_preview}')

    @staticmethod
    def return_bbox(sign):
        _, (x1, y1, x2, y2, score, class_id) = sign
        return x1, y1, x2, y2

    @staticmethod
    def check_direction(x1_1, w1, x1_2):
        if x1_1 - w1 * 0.1  < x1_2:
            return True
        return False

    def check_intersection(self, sign1, sign2):
        x1_1, y1_1, x2_1, y2_1 = sign1.get_bbox()
        x1_2, y1_2, x2_2, y2_2 = self.return_bbox(sign2)

        w1_ext = ((x2_1 - x1_1) * self.width_expansion_factor + abs(sign1.distance_delta[0]))*(2048 / y2_1)
        h1_ext = (y2_1 - y1_1) * self.height_expansion_factor + abs(sign1.distance_delta[1])
        w2 = (x2_2 - x1_2)
        h2 = (y2_2 - y1_2)

        if not self.check_direction(x1_1, w1_ext, x1_2):
            return 0

        x1_inter = max(x1_2, x1_1)
        y1_inter = max(y1_2, y1_1)
        x2_inter = min(x1_1 + w1_ext, x1_2 + w2)
        y2_inter = min(y1_1 + h1_ext, y1_2 + h2)

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        if self.debug_mode:
            image = self.curr_image.copy()
            if inter_area != 0:
                cv2.rectangle(image, (int(x1_1), int(y1_1)), (int(x1_1+w1_ext), int(y1_1+h1_ext)), (0, 0, 255), 3)
                cv2.rectangle(image, (int(x1_2), int(y1_2)), (int(x1_2+w2), int(y1_2+h2)), (0, 255, 255), 3)
                cv2.rectangle(image, (int(x1_1), int(y1_1)), (int(x2_1), int(y2_1)), (0, 255, 0), 4)
                cv2.rectangle(image, (int(x1_inter), int(y1_inter)), (int(x2_inter), int(y2_inter)), (255, 255, 0), 4)
                image = cv2.resize(image, (640, 640))
                cv2.imshow(f'debug_tracking', image)

        area_bbox2 = w2 * h2

        intersection_val =  inter_area / area_bbox2  if inter_area != 0 else 0
        return intersection_val

    def distance_between_centers(self, sign1, sign2):
        x1, y1, x1_, y1_ = sign1.get_bbox()
        x2, y2, x2_, y2_ = self.return_bbox(sign2)
        center1 = (x1 + x1_) / 2, (y1 + y1_) / 2
        center2 = (x2 + x2_) / 2, (y2 + y2_) / 2
        x_dist = center1[0] - center2[0]
        y_dist = center1[1] - center2[1]
        return x_dist, y_dist

    def add_sign(self, sign, image):
        _, (x1, y1, x2, y2, score, class_id) = sign
        if (score > self.score_threshold
                and sign[1][0] > self.image_size[0] - self.ROI[0]
                and sign[1][1] > self.image_size[1] - self.ROI[1]):
            self.signs[self.ID] = Sign(sign, self.ID, image)
            self.ID += 1

    def add_signs(self, signs, image):
        for sign in signs:
            self.add_sign(sign, image)

    def remove_signs(self, matched_indices, current_signs):
        selected_frames = []
        selected_signs = []
        selected_results = []
        for sign in current_signs:
            if sign.ID not in matched_indices:
                sign.increase_last_seen()
                if sign.last_seen > self.no_detection_threshold:
                    selected_signs.append(sign.return_sign())
                    selected_frames.append(sign.return_frame())
                    selected_results.append(sign.return_results())
                    del self.signs[sign.ID]
        return selected_frames, selected_signs, selected_results

    def outside_ROI(self, sign):
        x1, y1, x2, y2 = self.return_bbox(sign)
        if x2 > self.image_size[0] - self.ROI[2]:
            return True
        return False

    def match_signs(self, new_signs, current_signs, image):
        matched_indices = set()
        for new_sign in new_signs:
            best_match_sign = None
            best_match = 0

            if self.outside_ROI(new_sign):
                continue

            for sign in current_signs:
                actual_match = self.check_intersection(sign, new_sign)
                if actual_match > best_match:
                    best_match = actual_match
                    best_match_sign = sign

            if best_match > self.IOU_threshold:
                best_match_sign.update_sign(new_sign, self.distance_between_centers(best_match_sign, new_sign), image)
                matched_indices.add(best_match_sign.ID)
            else:
                self.add_sign(new_sign, image)

        return matched_indices

    def handle_tracking(self, new_sign_list, image):
        if self.debug_mode:
            self.curr_image = image
        if not self.signs:
            self.add_signs(new_sign_list, image)
            return [], [], [], image
        current_signs = list(self.signs.values())
        matched_indices = self.match_signs(new_sign_list, current_signs, image)
        selected_frames, selected_signs, selected_results = self.remove_signs(matched_indices, current_signs)
        if self.debug_mode:
            self.show_debug(self.curr_image)
        if self.enable_preview:
            image = self.create_preview(image)
        return selected_frames, selected_signs, selected_results, image

    def draw_bbox(self, image):
        for sign in self.signs.values():
            x1, y1, x2, y2 = sign.get_bbox()
            self.last_bbox = (x1, y1, x2, y2)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, f'{sign.ID}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 255, 0), 5)
        return image

    def create_preview(self, image_):
        image = image_.copy()
        image = self.draw_bbox(image)
        return image

    def show_debug(self, image_):
        image = image_.copy()
        cv2.rectangle(image, (self.image_size[0] - self.ROI[0], self.image_size[1] - self.ROI[1]),
                      (self.image_size[0], self.image_size[1]), (255, 0, 0), 2)
        if self.last_bbox:
            x1, y1, x2, y2 = self.last_bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        image = self.draw_bbox(image)

        image = cv2.resize(image, (640, 640))
        cv2.imshow('debug_tracker', image)
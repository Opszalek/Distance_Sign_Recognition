import cv2
from utils.utils import timeit


class Sign:
    def __init__(self, new_sign, ID):
        sign_img, (x1, y1, x2, y2, score, class_id) = new_sign
        self.bbox = (x1, y1, x2, y2)
        self.score = score
        self.class_id = class_id
        self.last_seen = 0
        self.ID = ID
        self.prev_images = []
        self.prev_images.append(sign_img)
        self.sign_img = sign_img

    def return_sign(self):
        return self.sign_img

    def return_results(self):
        return self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.score, self.class_id

    def compare_images(self, new_sign):
        sign_img, _ = new_sign
        for prev_image in self.prev_images:
            if cv2.absdiff(prev_image, sign_img).mean() < 10:
                return True
        return False

    def increase_last_seen(self):
        self.last_seen += 1

    def update_sign(self, new_sign):
        self.sign_img, (x1, y1, x2, y2, score, class_id) = new_sign
        self.bbox = (x1, y1, x2, y2)
        self.score = score
        self.last_seen = 0
        if self.class_id != class_id:
            raise ValueError('Class ID changed')  #TODO: add handling of class_id change

    def get_bbox(self):
        return self.bbox


class SignTracker:
    def __init__(self):
        self.signs = {}
        self.ID = 1
        self.no_detection_threshold = 3
        self.IOU_threshold = 0.6
        self.score_threshold = 0.6
        self.image_size = [2048, 2048]
        self.ROI = [1024, 1600, 60, 60]  #x1,y1,x_offset,y_offset

    @staticmethod
    def return_bbox(sign):
        _, (x1, y1, x2, y2, score, class_id) = sign
        return x1, y1, x2, y2

    def check_IOU(self, sign1, sign2):
        x1, y1, w1, h1 = sign1.get_bbox()
        x2, y2, w2, h2 = self.return_bbox(sign2)
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = w1 * h1
        boxBArea = w2 * h2
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def add_sign(self, sign):
        _, (x1, y1, x2, y2, score, class_id) = sign
        if (score > self.score_threshold
                and sign[1][0] > self.image_size[0] - self.ROI[0]
                and sign[1][1] > self.image_size[1] - self.ROI[1]):
            self.signs[self.ID] = Sign(sign, self.ID)
            self.ID += 1

    def add_signs(self, signs):
        for sign in signs:
            self.add_sign(sign)

    def remove_signs(self, matched_indices, current_signs):
        selected_signs = []
        selected_results = []
        for sign in current_signs:
            if sign.ID not in matched_indices:
                sign.increase_last_seen()
                if sign.last_seen > self.no_detection_threshold:
                    selected_signs.append(sign.return_sign())
                    selected_results.append(sign.return_results())
                    del self.signs[sign.ID]
        return selected_signs, selected_results

    def outside_ROI(self, sign):
        x1, y1, x2, y2 = self.return_bbox(sign)
        if x2 > self.image_size[0] - self.ROI[2]:
            return True
        return False

    def match_signs(self, new_signs, current_signs):
        matched_indices = set()
        for new_sign in new_signs:
            is_matched = False
            if self.outside_ROI(new_sign):
                continue
            for sign in current_signs:
                if self.check_IOU(sign, new_sign) > self.IOU_threshold:
                    sign.update_sign(new_sign)
                    matched_indices.add(sign.ID)
                    is_matched = True
                    break
            if not is_matched:
                self.add_sign(new_sign)
        return matched_indices

    def handle_tracking(self, new_sign_list):
        if not self.signs:
            self.add_signs(new_sign_list)
            return [], []
        current_signs = list(self.signs.values())
        matched_indices = self.match_signs(new_sign_list, current_signs)
        selected_signs, selected_results = self.remove_signs(matched_indices, current_signs)
        return selected_signs, selected_results

    def draw_bboxes(self, image_):
        image = image_.copy()
        cv2.rectangle(image, (self.image_size[0] - self.ROI[0], self.image_size[1] - self.ROI[1]),
                      (self.image_size[0], self.image_size[1]), (255, 0, 0), 2)
        for sign in self.signs.values():
            x1, y1, x2, y2 = sign.get_bbox()
            print(x1, y1)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, f'{sign.ID}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 0, 0), 5)
        image = cv2.resize(image, (640, 640))
        cv2.imshow('debug_image', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

import cv2
from utils.utils import timeit


class Sign:
    def __init__(self, new_sign, ID):
        sign_img, (x, y, w, h, score, class_id) = new_sign
        self.bbox = (x, y, w, h)
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
        sign_img, (x, y, w, h, score, class_id) = new_sign
        for prev_image in self.prev_images:
            if cv2.absdiff(prev_image, sign_img).mean() < 10:
                return True
        return False

    def increase_last_seen(self):
        self.last_seen += 1

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.last_seen = 0

    def update_sign(self, new_sign):
        self.sign_img, (x, y, w, h, score, class_id) = new_sign
        self.bbox = (x, y, w, h)
        self.score = score
        self.last_seen = 0
        if self.class_id != class_id:
            raise ValueError('Class ID changed')  #TODO: add handling of class_id change

    def get_bbox(self):
        return self.bbox


class SignTracker:
    def __init__(self):
        self.signs = {}
        # self.last_sign = None
        # self.current_sign_list = []
        self.ID = 1
        # self.no_detection_counter = 0
        self.no_detection_threshold = 3
        self.IOU_threshold = 0.6
        self.score_threshold = 0.6
        # self.ROI = [1024, 1024]

    @staticmethod
    def return_bbox(sign):
        _, (x, y, w, h, score, class_id) = sign
        return x, y, w, h

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
        _, (x, y, w, h, score, class_id) = sign
        if score > self.score_threshold:
            self.signs[self.ID] = Sign(sign, self.ID)
            self.ID += 1

    @timeit
    def handle_bboxes(self, new_sign_list):
        selected_signs = []
        selected_results = []
        current_signs = list(self.signs.values())
        if len(self.signs) == 0:
            for new_sign in new_sign_list:
                self.add_sign(new_sign)
        else:
            matched_indices = set()
            for new_sign in new_sign_list:
                is_matched = False
                for sign in current_signs:
                    #TODO: include roi in the calculation
                    if self.check_IOU(sign, new_sign) > self.IOU_threshold:
                        sign.update_sign(new_sign)
                        matched_indices.add(sign.ID)
                        is_matched = True
                        break
                if not is_matched:
                    self.add_sign(new_sign)
            for sign in current_signs:
                if sign.ID not in matched_indices:
                    sign.increase_last_seen()
                    if sign.last_seen > self.no_detection_threshold:
                        selected_signs.append(sign.return_sign())
                        selected_results.append(sign.return_results())
                        del self.signs[sign.ID]
        return selected_signs, selected_results

    def draw_bboxes(self, image_):
        image = image_.copy()
        for sign in self.signs.values():
            x, y, w, h = sign.get_bbox()
            cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 2)
            cv2.putText(image, f'{sign.ID}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 0, 0), 5)
        image = cv2.resize(image, (640, 640))
        cv2.imshow('debug_image', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

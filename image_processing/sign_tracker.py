class Sign:
    def __init__(self, new_sign, ID):
        sign_img, bbox, score, class_id = new_sign
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.last_seen = 0
        self.ID = ID
        self.prev_images = []
        self.prev_images.append(sign_img)

    def increase_last_seen(self):
        self.last_seen += 1

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.last_seen = 0

    def get_bbox(self):
        return self.bbox


class SignTracker:
    def __init__(self):
        self.signs = {}
        self.last_sign = None
        self.current_sign_list = []
        self.ID = 1
        self.no_detection_counter = 0
        self.no_detection_threshold = 3
        self.ROI = [1024, 1024]

    def check_IOU(self, sign1, sign2):
        #bboxes.append((x, y, w, h, score, class_id))
        #[(1210.884033203125, 1137.914306640625, 1297.21875, 1469.085205078125, 0.6613677740097046, 0.0)]
        x1, y1, w1, h1 = sign1.get_bbox()
        x2, y2, w2, h2 = sign2.get_bbox()
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = w1 * h1
        boxBArea = w2 * h2
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def handle_bboxes(self, new_sign_list):
        if len(new_sign_list) == 0:
            for sign in self.signs.values():
                sign.increase_last_seen()
                if sign.last_seen > self.no_detection_threshold:
                    #TODO: add function to remove sign that will return best image to pass to OCR
                    self.signs.pop(sign.ID)

        else:
            matched_indices = set()
            for new_sign in new_sign_list:
                for sign in self.signs.values():
                    iou = self.check_IOU(sign, new_sign)
                    if iou > 0.5:
                        sign.update_bbox(new_sign.get_bbox())
                        matched_indices.add(sign.ID)
                    else:
                        self.signs[self.ID] = Sign(new_sign, self.ID)
                        self.ID += 1

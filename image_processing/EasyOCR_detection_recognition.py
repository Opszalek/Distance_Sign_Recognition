# import easyocr
#
# class EasyOCR_sign:
#     def __init__(self):
#         self.reader = easyocr.Reader(['en'])
#
#     @staticmethod
#     def refactored_output(results):
#         if not results:
#             return None
#         refactored_result = []
#         for item in results:
#             if isinstance(item, tuple):
#                 bbox, text, confidence = item
#                 refactored_result.append([bbox, (text, confidence)])
#         return refactored_result
#
#     def predict_text(self, image):
#         result = self.reader.readtext(image, allowlist='0123456789')
#         return self.refactored_output(result)
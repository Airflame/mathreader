import cv2


class Processing:
    @staticmethod
    def extract_segments(image, draw_rectangles=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort()
        roi = [gray[y:y + height, x:x + width] for x, y, width, height in bounding_boxes]
        results = []
        for index, symbol in enumerate(roi):
            pixel = symbol[0, 0]
            if pixel == gray[0, 0]:
                if draw_rectangles:
                    x, y, width, height = bounding_boxes[index]
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                symbol = cv2.resize(symbol, (14, 21))
                results.append(symbol / 255)
        if draw_rectangles:
            cv2.imwrite("data/_output.png", image)
        return results
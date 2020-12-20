import cv2


class Processing:
    @staticmethod
    def extract_segments(image, draw_rectangles=False) -> list:
        """
        Divides an image into individual symbols and rescales them
        @param image: Image containing a mathematical expression
        @param draw_rectangles: Saves the image with rectangles drawn around contours if set to true
        @return: List containing symbols extracted from an image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, gray[0, 0] - 1, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(gray, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort()
        roi = [thresh[y:y + height, x:x + width] for x, y, width, height in bounding_boxes]
        results = []
        for index, symbol in enumerate(roi):
            x, y, width, height = bounding_boxes[index]
            if symbol[0, 0] == thresh[0, 0] or symbol[0, -1] == thresh[0, 0]:
                if draw_rectangles:
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                symbol = cv2.resize(symbol, (14, 21))
                results.append(symbol / 255)
        if draw_rectangles:
            cv2.imwrite("data/_output.png", image)
        return results

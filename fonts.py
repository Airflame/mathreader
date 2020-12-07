import cv2
from processing import Processing
from constants import Constants


class Fonts:
    def __init__(self, fonts):
        self.fonts = fonts
        self.data = []
        self.labels = [[0]*len(Constants.symbols) for _ in range(len(Constants.symbols))]

    def load(self) -> None:
        print("[ Loading " + str(self.fonts) + " fonts ]")
        for font in range(self.fonts):
            image = cv2.imread('data/font' + str(font) + '.png')
            self.data.extend(Processing.extract_segments(image))
        self.labels = [[0] * len(Constants.symbols) for _ in range(len(Constants.symbols))]
        for i in range(len(self.labels)):
            self.labels[i][i] = 1
        self.labels *= self.fonts

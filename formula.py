import cv2
from network import Network
from processing import Processing
from constants import Constants


class Formula:
    def __init__(self):
        self.solved = ""
        self.image = None
        self.segments = []
        self.formula = ""

    def load(self, filename) -> None:
        # print("( Loading formula from file " + filename + " )")
        self.image = cv2.imread(filename)
        self.segments = Processing.extract_segments(self.image)

    def evaluate(self, network: Network):
        # print("( Evaluating formula using neural network )")
        for segment in self.segments:
            self.formula += Constants.symbols[network.evaluate(input_vector=segment)]
        # print(self.formula)
        # print(eval(self.formula))
        self.solved = self.formula
        self.solved += (" = " + str(eval(self.formula)))

        return self.solved

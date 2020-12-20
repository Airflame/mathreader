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
        """
        Loads an image containing a mathematical expression and extracts individual symbols from it
        @param filename: Path to a file
        """
        self.image = cv2.imread(filename)
        self.segments = Processing.extract_segments(self.image, draw_rectangles=True)

    def evaluate(self, network: Network) -> str:
        """
        Uses a neural network to recognize symbols and calculate the result
        @param network: Network used for evaluation
        @return: Solved formula
        """
        self.formula = ""
        for segment in self.segments:
            self.formula += Constants.symbols[network.evaluate(input_vector=segment)]
        self.solved = self.formula
        try:
            self.solved += (" = " + str(eval(self.formula)))
        except:
            self.solved += " = NaN"
        return self.solved

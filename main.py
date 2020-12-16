from fonts import Fonts
from network import Network
from formula import Formula
from constants import Constants
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QIcon
import numpy as np


class MathReader(QWidget):
    def __init__(self):
        super().__init__()
        self.iterations = 10000
        self.fonts = Fonts(2)
        fonts_used = self.fonts.fonts
        self.fonts.load()
        self.neurons = (35, 25, len(Constants.symbols))
        self.network = Network(neurons=self.neurons, input_size=21 * 14)
        self.formula = Formula()
        self.file_path = ""

        self.setWindowTitle("MathReader")
        self.setWindowIcon(QIcon('icon.png'))
        grid = QGridLayout()
        self.setGeometry(500, 200, 730, 500)

        label_math_reader = QLabel("MathReader")
        label_math_reader.setFont(QFont("Ink Free", 20, QFont.Bold))
        label_math_reader.setMaximumHeight(50)
        label_math_reader.setAlignment(Qt.AlignCenter)

        btn_save_weights = QPushButton("Save weights")
        btn_save_weights.clicked.connect(self.save_weights)
        btn_load_weights = QPushButton("Load weights")
        btn_load_weights.clicked.connect(self.load_weights)
        btn_train = QPushButton("Train")
        btn_train.clicked.connect(self.training)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.open_file)
        btn_solve = QPushButton("Solve!")
        btn_solve.clicked.connect(self.solve_formula)

        label_file = QLabel("Chosen image directory: ")
        label_open_image = QLabel("Use the 'Browse' button to select graphic file, you would like me to read from :)")
        label_open_image.setAlignment(Qt.AlignCenter)
        self.label_image = QLabel("Your graphic file will load here!")
        self.label_image.setStyleSheet("background-color: white")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.browsed_file = QLineEdit()
        self.browsed_file.setMaximumWidth(300)
        self.solved_formula = QLineEdit()
        label_solved = QLabel("I think it is...")
        label_solved.setAlignment(Qt.AlignBottom)

        start_state = "[ Loading " + str(fonts_used) + " fonts ]"
        start_state += "\n{ Initializing network with layers of sizes " + str(self.neurons) + " }"
        self.label_state = QLabel(start_state)
        self.label_state.setStyleSheet("background-color: white")
        self.label_state.setAlignment(Qt.AlignCenter)

        grid.addWidget(label_math_reader, 0, 0, 1, 1)
        grid.addWidget(btn_train, 1, 0)
        grid.addWidget(btn_save_weights, 2, 0)
        grid.addWidget(btn_load_weights, 3, 0)
        grid.addWidget(btn_solve, 5, 0)

        grid.addWidget(label_open_image, 0, 2, 1, 3)
        grid.addWidget(label_file, 1, 2, 1, 1)
        grid.addWidget(self.browsed_file, 1, 3, 1, 1)
        grid.addWidget(btn_browse, 1, 4, 1, 1)
        grid.addWidget(self.label_image, 2, 2, 7, 3)

        grid.addWidget(label_solved, 9, 0, 1, 1)
        grid.addWidget(self.solved_formula, 10, 0, 1, 5)
        grid.addWidget(self.label_state, 11, 0, 1, 5)
        grid.setSpacing(10)
        self.setLayout(grid)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.browsed_file.setText(file_name)
        self.file_path = file_name
        self.label_image.setPixmap(QPixmap(file_name))
        self.label_state.setText("Loaded formula from file " + file_name)

    def training(self):
        self.network.fit(iterations=self.iterations, input_data=self.fonts.data, input_labels=self.fonts.labels)
        train_state = "Trained network for " + str(self.iterations) + " iterations and " + \
                      str(len(self.fonts.data)) + " samples"
        train_state += "\nNetwork successfully trained."
        self.label_state.setText(train_state)

    def save_weights(self):
        self.network.save("weights")
        self.label_state.setText("Saving weights to file weights.csv")

    def load_weights(self):
        self.network.load("weights")
        self.label_state.setText("Loading weights from file weights.csv")

    def solve_formula(self):
        self.label_state.setText("Evaluating formula using neural network")
        self.formula.load(self.file_path)
        self.solved_formula.setText(self.formula.evaluate(self.network))


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MathReader()
    window.show()
    sys.exit(app.exec_())

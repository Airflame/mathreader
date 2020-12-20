import threading
from fonts import Fonts
from network import Network
from formula import Formula
from constants import Constants
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QProgressBar
from PyQt5.QtWidgets import QLabel, QLineEdit, QFileDialog, QFrame, QDialog
from PyQt5.QtGui import QPixmap, QFont, QIcon


class MathReader(QWidget):
    def __init__(self):
        super().__init__()
        self.iterations = 10000
        self.ro = 0.5
        self.alpha = 0.5
        self.fonts = Fonts(3)
        fonts_used = self.fonts.fonts
        self.fonts.load()
        self.network = Network(neurons=(35, 25, len(Constants.symbols)), input_size=21 * 14)
        self.formula = Formula()
        self.file_path = ""
        self.network_ready = False

        self.setWindowTitle("MathReader")
        self.setWindowIcon(QIcon('res/icon.png'))
        grid = QGridLayout()
        self.setFixedSize(730, 500)

        label_math_reader = QLabel("MathReader")
        label_math_reader.setFont(QFont("Ink Free", 20, QFont.Bold))
        label_math_reader.setMaximumHeight(50)
        label_math_reader.setAlignment(Qt.AlignCenter)

        self.btn_save_weights = QPushButton("Save weights")
        self.btn_save_weights.clicked.connect(self.save_weights)
        self.btn_load_weights = QPushButton("Load weights")
        self.btn_load_weights.clicked.connect(self.load_weights)
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self.open_training_dialog)
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.open_file)
        self.btn_solve = QPushButton("Solve!")
        self.btn_solve.setEnabled(False)
        self.btn_solve.clicked.connect(self.solve_formula)

        label_file = QLabel("Chosen image directory: ")
        label_open_image = QLabel("Use the 'Browse' button to select graphic file, you would like me to read from :)")
        label_open_image.setAlignment(Qt.AlignCenter)
        self.label_image = QLabel("Your graphic file will load here!")
        self.label_image.setStyleSheet("background-color: white")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.browsed_file = QLineEdit()
        self.browsed_file.setReadOnly(True)
        self.browsed_file.setMaximumWidth(300)
        self.solved_formula = QLineEdit()
        self.solved_formula.setReadOnly(True)
        self.solved_formula.setFont(QFont("Bahnschrift", 12))
        self.solved_formula.setAlignment(Qt.AlignCenter)
        label_solved = QLabel("I think it is...")
        label_solved.setMaximumHeight(20)
        label_solved.setAlignment(Qt.AlignBottom)
        self.loading = QProgressBar()
        label_loading = QLabel("Training state: ")

        self.separator_h = QFrame()
        self.separator_h.setFrameShape(QFrame.HLine)
        self.separator_h.setFrameShadow(QFrame.Sunken)
        separator_v = QFrame()
        separator_v.setFrameShape(QFrame.VLine)
        separator_v.setFrameShadow(QFrame.Sunken)

        start_state = "Loaded " + str(fonts_used) + " fonts"
        start_state += ("\nInitializing network with layers of sizes " + str(self.network.neurons))
        self.label_state = QLabel(start_state)
        self.label_state.setStyleSheet("background-color: white")
        self.label_state.setAlignment(Qt.AlignCenter)

        self.training_dialog = QDialog()
        self.training_params = QLabel("Here you can specify training parameters: ")
        self.choose_iterations_label = QLabel("Iterations number: ")
        self.choose_alpha_label = QLabel("\u03B1 value: ")
        self.choose_rho_label = QLabel("\u03C1 value: ")
        self.choose_iterations_input = QLineEdit()
        self.choose_iterations_input.setText("10000")
        self.choose_alpha_input = QLineEdit()
        self.choose_alpha_input.setText("0.5")
        self.choose_rho_input = QLineEdit()
        self.choose_rho_input.setText("0.5")
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setMaximumWidth(80)

        grid.addWidget(label_math_reader, 0, 0, 1, 1)
        grid.addWidget(self.btn_train, 1, 0)
        grid.addWidget(self.btn_save_weights, 2, 0)
        grid.addWidget(self.btn_load_weights, 3, 0)
        grid.addWidget(self.btn_solve, 5, 0)
        grid.addWidget(separator_v, 0, 1, 10, 1)

        grid.addWidget(label_open_image, 0, 2, 1, 3)
        grid.addWidget(label_file, 1, 2, 1, 1)
        grid.addWidget(self.browsed_file, 1, 3, 1, 1)
        grid.addWidget(self.btn_browse, 1, 4, 1, 1)
        grid.addWidget(self.label_image, 2, 2, 7, 3)
        grid.addWidget(label_loading, 9, 2, Qt.AlignCenter)
        grid.addWidget(self.loading, 9, 3, 1, 2)

        grid.addWidget(label_solved, 10, 0, 1, 1)
        grid.addWidget(self.solved_formula, 11, 0, 1, 5)
        grid.addWidget(self.label_state, 12, 0, 1, 5)
        grid.setSpacing(10)
        self.setLayout(grid)

        self.training_dialog.setModal(True)
        self.training_dialog.setWindowTitle("Training Dialog")
        self.training_dialog.setWindowIcon(QIcon('res/icon.png'))
        grid = QGridLayout()
        self.btn_apply.clicked.connect(self.get_training_params)
        self.training_dialog.setFixedSize(230, 150)

        grid.addWidget(self.training_params, 0, 0, 1, 2)
        grid.addWidget(self.separator_h, 1, 0, 1, 2)
        grid.addWidget(self.choose_iterations_label, 2, 0)
        grid.addWidget(self.choose_iterations_input, 2, 1)
        grid.addWidget(self.choose_alpha_label, 3, 0)
        grid.addWidget(self.choose_alpha_input, 3, 1)
        grid.addWidget(self.choose_rho_label, 4, 0)
        grid.addWidget(self.choose_rho_input, 4, 1)
        grid.addWidget(self.btn_apply, 5, 0, 1, 2, Qt.AlignCenter)

        self.training_dialog.setLayout(grid)

    def open_file(self) -> None:
        """
        Opens a file dialog used to load image with a mathematical expression
        """
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file')
        self.browsed_file.setText(file_name)
        self.file_path = file_name
        self.w = min(530, QPixmap(file_name).width())
        self.h = min(270, QPixmap(file_name).height())
        self.label_image.setPixmap(QPixmap(file_name).scaled(self.w, self.h, Qt.KeepAspectRatio))
        self.label_state.setText("Loaded formula from file " + file_name)
        self.btn_solve.setEnabled(True)

    def save_weights(self) -> None:
        """
        Opens a file dialog used to save network parameters to a .csv file
        """
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save file', filter="csv(*.csv)")
        if file_name != '':
            file = open(file_name, 'w')
            with open(file_name):
                self.network.save(file_name)
            file.close()
            self.label_state.setText("Saving weights to file " + file_name)

    def load_weights(self) -> None:
        """
        Opens a file dialog used to load network parameters from a .csv file
        """
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open file', filter="csv(*.csv)")
        if file_name != '':
            self.network.load(file_name)
            self.label_state.setText("Loaded weights from file " + file_name)
            self.network_ready = True

    def solve_formula(self) -> None:
        """
        Solves the formula recognized in the image by the neural network
        """
        if self.network_ready:
            self.label_state.setText("Evaluating formula using neural network.")
            self.formula.load(self.file_path)
            self.solved_formula.setText(self.formula.evaluate(self.network))
            self.label_image.setPixmap(QPixmap("data/_output.png").scaled(self.w, self.h, Qt.KeepAspectRatio))
        else:
            self.label_state.setText("Please train network or load weights first.")

    def open_training_dialog(self) -> None:
        """
        Opens the dialog before training in which the user sets the number of iterations and additional parameters.
        """
        self.training_dialog.exec()

    def get_training_params(self) -> None:
        """
        Retrieves the training parameters from closed dialog and starts training with
        """
        self.iterations = int(self.choose_iterations_input.text())
        self.ro = float(self.choose_rho_input.text())
        self.alpha = float(self.choose_alpha_input.text())
        self.training_dialog.close()
        t = threading.Thread(target=self.network.fit, args=(self.iterations,
                                                                        self.fonts.data, self.fonts.labels,
                                                                        self.ro, self.alpha, self.set_progress_bar))
        t.start()
        self.label_state.setText("Training network for " + str(self.iterations) + " iterations and " +
                                 str(len(self.fonts.data)) + " samples.")
        self.network_ready = True

    def set_progress_bar(self, progress) -> None:
        """
        Updates the training progress bar with data from neural network
        @param progress: Integer in range from 0 to 100 representing training progress in percentage
        """
        self.loading.setValue(round(float(progress)))
        value = self.loading.value()
        if value < 100:
            self.btn_train.setEnabled(False)
            self.btn_load_weights.setEnabled(False)
            self.btn_save_weights.setEnabled(False)
            self.btn_browse.setEnabled(False)
        else:
            self.label_state.setText("Network successfully trained.")
            self.btn_train.setEnabled(True)
            self.btn_load_weights.setEnabled(True)
            self.btn_save_weights.setEnabled(True)
            self.btn_browse.setEnabled(True)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MathReader()
    window.show()
    sys.exit(app.exec_())

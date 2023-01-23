from PyQt5.QtWidgets import*
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

class grafik(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axis1 = self.canvas.figure.add_subplot(111)
        # self.canvas.figure.set_facecolor("xkcd:beige")
        self.setLayout(vertical_layout)

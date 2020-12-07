from fonts import Fonts
from network import Network
from formula import Formula
from constants import Constants

fonts = Fonts(2)
fonts.load()

network = Network(neurons=(35, 25, len(Constants.symbols)), input_size=21*14)
#network.load("weights")
network.fit(iterations=10000, input_data=fonts.data, input_labels=fonts.labels)
#network.save("weights")

formula = Formula()
formula.load('data/formula.png')
print(formula.evaluate(network))


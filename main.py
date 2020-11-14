import processing
from network import Network
import numpy as np
import matplotlib.pyplot as plt
import cv2

symbols = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")")
fonts = 2

E = []
for font in range(fonts):
    image = cv2.imread('data/font' + str(font) + '.png')
    E.extend(processing.extract_symbols(image))

C = [[0]*len(symbols) for _ in range(len(symbols))]
for i in range(len(C)):
    C[i][i] = 1
C *= fonts

network = Network(neurons=(35, 25, len(symbols)), input_size=21*14)
network.load("weights")
#network.fit(iterations=20000, input_data=E, input_labels=C)
#network.save("weights")

#R = [E[5], E[10], E[3], E[10], E[2]]
#R = E
image = cv2.imread('data/formula.png')
R = processing.extract_symbols(image)

formula = ""
for i in range(len(R)):
    plt.imshow(np.array(R[i]).reshape(21, 14))
    plt.show()
    formula += symbols[network.evaluate(input_vector=R[i])]

# plt.imshow(np.array(img).reshape(21, 14))
# plt.show()
print(formula+" -> "+str(eval(formula)))


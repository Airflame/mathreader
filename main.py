from network import Network
import numpy as np
import matplotlib.pyplot as plt
import cv2

symbols = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")")
fonts = 2

E = []
for font in range(fonts):
    image = cv2.imread('data/fontsy' + str(font) + '.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    boundingBoxes.sort()

    max_height = 0
    max_width = 0
    for box in boundingBoxes:
        x, y, width, height = box
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

    print(boundingBoxes)
    roi = [gray[y:y + height, x:x + width] for x, y, width, height in boundingBoxes]

    for symbol in roi:
        pixel = symbol[0, 0]
        if pixel == 255:
            result = np.full((max_height, max_width, 1), 255, dtype=np.uint8)
            result[0:symbol.shape[0], 0:symbol.shape[1], 0] = symbol
            result = cv2.resize(result, (14, 21))
            E.append(result/255)

C = [[0]*len(symbols) for _ in range(len(symbols))]
for i in range(len(C)):
    C[i][i] = 1
C *= fonts

network = Network(neurons=(35, 25, len(symbols)), input_size=21*14)
#network.load("weights")
network.fit(iterations=20000, input_data=E, input_labels=C)

# R = [E[5], E[10], E[3], E[10], E[2]]
R = E

formula = ""
for i in range(len(R)):
    plt.imshow(np.array(R[i]).reshape(21, 14))
    plt.show()
    formula += symbols[network.evaluate(input_vector=R[i])]

# plt.imshow(np.array(img).reshape(21, 14))
# plt.show()
print(formula+" -> +str(eval(formula))")


from network import Network
import numpy as np
import matplotlib.pyplot as plt
import cv2

symbols = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "=", "(", ")")
fonts = 5

E = []
for font in range(fonts):
    for symbol in range(len(symbols)):
        img = cv2.cvtColor(cv2.imread('data/'+str(font)+'-'+str(symbol)+'.png'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (14, 21))
        E.append(img/255)

C = [[0]*17 for _ in range(17)]
for i in range(len(C)):
    C[i][i] = 1
C *= fonts

network = Network(neurons=(35, 25, 17), input_size=21*14)
#network.load("weights")
network.fit(iterations=30000, input_data=E, input_labels=C)

# R = [E[5], E[10], E[3], E[10], E[2]]
R = E

formula = ""
for i in range(len(R)):
    # plt.imshow(np.array(R[i]).reshape(21, 14))
    # plt.show()
    formula += symbols[network.evaluate(input_vector=R[i])]

# plt.imshow(np.array(img).reshape(21, 14))
# plt.show()
print(formula+" -> +str(eval(formula))")

img = cv2.cvtColor(cv2.imread('data/resize6.png'), cv2.COLOR_BGR2GRAY) / 255
img = cv2.resize(img, (14, 21))
plt.imshow(np.array(img).reshape(21, 14))
plt.show()
print(symbols[network.evaluate(input_vector=img)])

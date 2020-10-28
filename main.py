from network import Network
import numpy as np
import matplotlib.pyplot as plt
import cv2

symbols = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "=", "(", ")")

E = []
for i in range(len(symbols)):
    img = cv2.cvtColor(cv2.imread('data/'+str(i)+'.png'), cv2.COLOR_BGR2GRAY)
    E.append(img/255)

C = [[0]*17 for _ in range(17)]

for i in range(len(C)):
    C[i][i] = 1

network = Network(neurons=(35, 25, 17), input_size=21*14)
network.fit(iterations=10000, input_data=E, input_labels=C)

#R = [E[5], E[10], E[3], E[10], E[2]]
img = cv2.cvtColor(cv2.imread('data/test3.png'), cv2.COLOR_BGR2GRAY) / 255

formula = ""
# for i in range(len(R)):
#     plt.imshow(np.array(R[i]).reshape(21, 14))
#     plt.show()
#     formula += symbols[network.evaluate(input_vector=R[i])]

plt.imshow(np.array(img).reshape(21, 14))
plt.show()
formula += symbols[network.evaluate(input_vector=img)]
print(formula+" -> "+str(eval(formula)))

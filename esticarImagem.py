import LeituraOmni as omni
import numpy as np
import cv2
from time import time

t0 = time()

normal, esticada, yolo = omni.esticarImagem('imagem2.jpg', (988, 988), (0, 0))

print(time()-t0)

cv2.imshow('Atual', normal)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Atual', esticada)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Atual', yolo)
cv2.waitKey(0)
cv2.destroyAllWindows()

h = esticada.shape[0]
l = esticada.shape[1]

branco = np.ones((1, h*l*3))
branco = np.reshape(branco, esticada.shape)
branco = cv2.resize(branco, (0,0), fx=1, fy=0.2)*255

normalr = cv2.resize(normal, (0,0), fx=l/normal.shape[0], fy=l/normal.shape[0])
esticadar = cv2.resize(esticada, (0,0), fx=1, fy=1.7)
yolor = cv2.resize(yolo, (0,0), fx=1, fy=1.7)

img1 = np.concatenate((normalr, branco), axis = 0)
img2 = np.concatenate((img1, esticadar), axis = 0)
img3 = np.concatenate((img2, branco), axis = 0)
im4 = np.concatenate((img3, yolor), axis = 0)

cv2.imwrite('concatenada.png', im4, [cv2.IMWRITE_JPEG_QUALITY, 100])
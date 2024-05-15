import LeituraOmni as omni
import cv2
from time import time

tempo = time()
print(tempo)
    
parametrosOmni = omni.inicializarOmni(1)

tempo2 = time()
print(f"\n{tempo}\n{tempo2}")

while 1:
    frame, resultado = omni.lerOmni(parametrosOmni)
    frame = cv2.resize(frame, (int(parametrosOmni.resolucao[0]*2/3), int(parametrosOmni.resolucao[1]*1/3)))
    cv2.imshow("resultado", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    
from ultralytics import YOLO
import numpy as np
import cv2
import time

class criarParametrosOmni:
    def __init__(self, resolucao, ajusteCentro, corteInf, corteSup, angSup, angInf, centro, tamanhoFrame, videoFinal, intervalo, model, uimgc, vimgc, cap):
        # Parametros de ajuste
        self.resolucao = resolucao
        self.ajusteCentro = ajusteCentro
        self.corteInf = corteInf
        self.corteSup = corteSup
        self.angSup = angSup
        self.angInf = angInf
        self.centro = centro
        self.tamanhoFrame = tamanhoFrame
        self.videoFinal = videoFinal
        self.intervalo = intervalo
        self.model = model
        self.uimgc = uimgc
        self.vimgc = vimgc
        self.cap = cap
        self.tempo = time.time()

def _esticarHiperbolicamente(tamanhoFrame, angSup, angInf, corteInf, corteSup):
    rCilindro = 500
    angInf = np.deg2rad(angInf)
    angSup = np.deg2rad(angSup)

    Vpn = np.tan(abs(angInf))*rCilindro + np.tan(abs(angSup))*rCilindro
    Hpn = int(Vpn*2*np.pi) # Comprimento da imagem nova
    hInf = corteInf
    hSup = tamanhoFrame-corteSup
    
    uvect = np.arange(0, Hpn, 1)
    vvect = np.arange(0, Vpn, 1)
    upn, vpn = np.meshgrid(uvect, vvect)

    r = ((np.arctanh((vpn-(np.tan(abs(angInf))*rCilindro))/rCilindro))-angInf)/(angSup-angInf)*(hSup-hInf)+hInf

    uimg = (r*np.cos(upn*2*np.pi/Hpn)+tamanhoFrame).astype(int)
    vimg = (-r*np.sin(upn*2*np.pi/Hpn)-tamanhoFrame).astype(int)

    return (uimg, vimg)

def _esticarConicamente(tamanhoFrame):
    rCIlindro = 219

    # Calculos de parametros da transformação
    altura = tamanhoFrame*2
    rpixel = int(altura/2) # definição do raio da imagem original
    Vpn = int(altura/2) # Altura da imagem
    Hpn = int(Vpn*2*np.pi) # Comprimento da imagem nova

    # Criação das matrizes para o calculo
    uvect = np.arange(0, Hpn, 1) #Vetor de coordenadas em x
    vvect = np.arange(0, Vpn, 1) #Vetor de coordenadas em y
    upn, vpn = np.meshgrid(uvect, vvect) #matriz de coordenadasd vetorizadas

    # Transformação de coordenadas
    uimg = (((vpn*rpixel)/Vpn)*np.cos((upn*2*np.pi)/Hpn) + rpixel). astype(int)
    vimg = (-(((vpn*rpixel)/Vpn)*np.sin((upn*2*np.pi)/Hpn) - rpixel)). astype(int)

    return (uimg, vimg)

def inicializarOmni(portaCamera, resolucao = (1920, 1080), ajusteCentro = (0, 40), corteInf = 117, corteSup = 1, gravarVideo = 0, intervalo = -1):
    """
    inicializa a camera, criando os parametros necessários
    "gravarVideo" valor booleano para decidir se quer gravar o video da camera
    "intervalo" intervalo, em segundos, entre as fotos salvas, -1 para não tirar fotos
    """

    print("Inicializando Omni...")

    # Parametros de ajuste
    # resolucao = (1920, 1080)
    # ajusteCentro = (0, 40) # ajuste da imagem salva = (0, -14)
    # corteInf = 117
    # corteSup = 1
    angSup = 15
    angInf = 20

    # Variáveis de ajuste da imagem
    centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
    tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))

    if gravarVideo:
        videoFinal = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, resolucao)
    else:
        videoFinal = ""


    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8m.pt')

    # Tenta abrir o CSV com as lookup tables, se não cria uma nova e salva
    try:
        uimgc = np.loadtxt('uimgc.csv').astype(np.int64)
        vimgc = np.loadtxt('vimgc.csv').astype(np.int64)

    except:
        uimgc, vimgc = _esticarConicamente(tamanhoFrame)
        np.savetxt('uimgc.csv', uimgc)
        np.savetxt('vimgc.csv', vimgc)


    # Configurações da camera
    cap = cv2.VideoCapture(portaCamera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Não é possível abrir a camera!")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao[1])
    cap.set(cv2.CAP_PROP_FPS, 60)

    parametrosOmni = criarParametrosOmni(resolucao, ajusteCentro, corteInf, corteSup, angSup, angInf, centro, tamanhoFrame, videoFinal, intervalo, model, uimgc, vimgc, cap)

    print("Omni inicializada com sucesso!")
    return(parametrosOmni)

def lerOmni(parametrosOmni):
    """
    Le uma imagem hiperbólica com a biblioteca openCV2 e retorna a imagem esticada
    """
    # Captura o frame
    ret, frame = parametrosOmni.cap.read()

    # Ajustes da imagem
    img = frame[parametrosOmni.centro[1]-parametrosOmni.tamanhoFrame:parametrosOmni.centro[1]+parametrosOmni.tamanhoFrame, parametrosOmni.centro[0]-parametrosOmni.tamanhoFrame:parametrosOmni.centro[0]+parametrosOmni.tamanhoFrame] #Transformnado para geometria quadrada e centralizando oespelho

    # Transformação conica
    imgFinalc = np.flip(img, 1)
    imgFinalc = np.copy(imgFinalc[parametrosOmni.vimgc, parametrosOmni.uimgc]) #Salvando os pixels ads coordenadas tranformadas na imagem final
    imgFinalc = np.flip(imgFinalc, 0)
    imgFinalc = imgFinalc[parametrosOmni.corteSup:-parametrosOmni.corteInf, :]

    # Salva a foto a cada 2 segundos
    if (time.time() - parametrosOmni.tempo) > parametrosOmni.intervalo >=0:
        frame_name = './frame' + str(time.time() - parametrosOmni.tempo) + '.jpg'
        parametrosOmni.tempo = time.time()
        cv2.imwrite(frame_name, imgFinalc, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print("Click!")

    # Grava o frame atual em videoFinal
    if parametrosOmni.videoFinal != "":
        parametrosOmni.videoFinal.write(imgFinalc)

    res = parametrosOmni.model(imgFinalc, verbose = False)
    res_plotted = res[0].plot()
    imgFinal = np.concatenate((res_plotted, imgFinalc), axis = 0)
    return(imgFinal, res)

def esticarImagem(caminho, resolucao = (1920, 1080), ajusteCentro = (0, -14), corteInf = 117, corteSup = 1, YOLOescolha = 1):
    """
    Inicializa as variaveis necessárias para ler uma imagem ja salva no caminho indicado
    """

    print("Inicializando Omni...")

    # Parametros de ajuste
    # resolucao = (1920, 1080)
    # ajusteCentro = (0, 40) # ajuste da imagem salva = (0, -14)
    # corteInf = 117
    # corteSup = 1
    angSup = 15
    angInf = 20
    resolucaoFinal = 30/64

    # Variáveis de ajuste da imagem
    centro = (resolucao[0]//2 + ajusteCentro[0], resolucao[1]//2 + ajusteCentro[1])
    tamanhoFrame = (resolucao[1]//2-abs(ajusteCentro[1]))

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8m.pt')

    # Tenta abrir o CSV com as lookup tables, se não cria uma nova e salva
    try:
        uimgc = np.loadtxt('uimgc.csv').astype(np.int64)
        vimgc = np.loadtxt('vimgc.csv').astype(np.int64)

    except:
        uimgc, vimgc = _esticarConicamente(tamanhoFrame)
        np.savetxt('uimgc.csv', uimgc)
        np.savetxt('vimgc.csv', vimgc)

    frame = cv2.imread(caminho)

    # Corta e centraliza
    if resolucao[0] != resolucao[1]:
        frame = frame[centro[1]-tamanhoFrame:centro[1]+tamanhoFrame, centro[0]-tamanhoFrame:centro[0]+tamanhoFrame] #Transformnado para geometria quadrada e centralizando oespelho

    # Transformação conica
    imgFinalc = np.copy(frame[vimgc, uimgc]) #Salvando os pixels ads coordenadas tranformadas na imagem final
    imgFinalc = np.flip(imgFinalc, 0)
    imgFinalc = imgFinalc[corteSup:-corteInf, :]
    imgFinalc = cv2.resize(imgFinalc, (int(imgFinalc.shape[1]*resolucaoFinal), int(imgFinalc.shape[0]*resolucaoFinal)))

    if YOLOescolha:
        res = model(imgFinalc, imgsz=imgFinalc.shape[0:-1], iou=0.5, conf=0.1)
        res_plotted = res[0].plot(show_labels=False, show_conf=False)
        #cv2.imshow("resultado", res_plotted/255)
        imgFinal = np.concatenate((res_plotted, imgFinalc), axis = 0)
        cv2.imwrite(f'{caminho}EsticadaYOLO.png', imgFinal, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        #cv2.imshow("resultado", imgFinalc)
        cv2.imwrite(f'{caminho}Esticada.png', imgFinalc, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Omni inicializada com sucesso!")

    return(frame, imgFinalc, res_plotted)

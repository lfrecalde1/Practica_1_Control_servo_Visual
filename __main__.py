#!/home/fer/.virtualenvs/cv/bin/python
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA

## Colocar las direcciones donde se encuentran las fotografias
path_o = '/home/fer/Control_servo_visual/Code/Practico_1.0/Pictures/'
path_w = '/home/fer/Control_servo_visual/Code/Practico_1.0/Modificadas/'

## Funcion para la lectura de la imagenes del sistema
def data(path,aux=1):
    images = []
    index = os.listdir(path)
    index.sort()
    for img in index:
        pictures = cv2.imread(os.path.join(path,img), aux)
        images.append([pictures])
    if aux==1:
        img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1],pictures.shape[2])
    else:
        img = np.array(images,dtype=np.uint8).reshape(len(images),pictures.shape[0],pictures.shape[1])
    return img

def mapeo(img, f):
    aux = len(img.shape)
    if aux ==3:
        b, g, r = cv2.split(img)
        ## Creacion de las matrices vacias de la funcion
        B = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
        G = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
        R = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)

        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B[i,j] = f(b[i,j]) 
                G[i,j] = f(g[i,j])
                R[i,j] = f(r[i,j])
        new = cv2.merge([B, G, R])

    else:
        new = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                new[i,j] = f(img[i,j])
    return new

def mapeo_d(img, f):
    aux = len(img.shape)
    if aux ==3:
        b, g, r = cv2.split(img)
        ## Creacion de las matrices vacias de la funcion
        B = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        G = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        R = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)

        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B[i,j] = f(b[i,j]) 
                G[i,j] = f(g[i,j])
                R[i,j] = f(r[i,j])
        new = cv2.merge([B, G, R])

    else:
        new = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                new[i,j] = f(img[i,j])
    return new
def show(img, new):
    ## Mostrar las imagenes por pantalla
    cv2.imshow('Normal', img)
    cv2.imshow('Modificada', new)
    cv2.waitKey(0)
    return None

def guardar(new, name):
    cv2.imwrite(os.path.join(path_w,name), new)
    return None


def inver_o(pixel):
    f = 255-pixel
    return f

## Funcion para 255-p(x,y)
def inver(img,contador):
    new = mapeo(img,inver_o)
    name = "Pregunta_1_{}.png".format(contador)
    show(img, new)
    guardar(new, name)
    return None

def rango(img,contador):
    new = mapeo(img, rango_o)
    name = "Pregunta_2_{}.png".format(contador)
    show(img, new)
    guardar(new, name)

    return None

def rango_o(pixel, a=100, b=200):
    ## funcion para el rango
    if 0<=pixel<a:
        f = 0
    elif a<=pixel<=b:
        f=pixel
    elif pixel>b:
        f=255
    return f

def gamma_o(pixel, gama = 1.5):
    valor = 1/gama
    f = np.power(pixel, valor)
    return f

def gamma(img,contador):
    img =img/255
    new = mapeo_d(img, gamma_o)
    Final = cv2.normalize(new, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ## Nombre de los archivos a ser guardados
    name = "Pregunta_3_{}.png".format(contador)
    show(img, new)
    guardar(Final, name)
    return None

def linear_tranfor(pixel, alpha = 1.5, beta = 50):
    f = alpha*pixel+beta
    if f>255:
        f=255
    elif f<0:
        f=0
    return f

def basic_linear_transform(img, contador):
    new = mapeo(img, linear_tranfor)
    name = "Pregunta_4_{}.png".format(contador)
    show(img, new)
    guardar(new, name)
    return None

def cal_histograma(img):
    aux = len(img.shape)
    if aux ==3:
        color = ('b', 'g', 'r')
        histograma = np.zeros((256,3))
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0,256])
            histograma[:,i]=histr.reshape((256))
    else:
        color = ('b')
        histograma = np.zeros((256,3))
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0,256])
            histograma[:,i]=histr.reshape((256))

    return histograma,color


def grafica(histr, modificada ,color):
    fig = plt.figure('Histogramas',figsize=(8,5),tight_layout=True)
    axs = fig.subplots(2,1,sharex=True)
    #fig, axs = plt.subplots(2, 1, constrained_layout=True)
    for i,col in enumerate(color):
        axs[0].plot(histr[:,i], color = col, label = col)
        axs[0].set_xlabel(r'Valor pixel')
        axs[0].set_ylabel(r'Frecuencia Pixel')
        #axs[0].set_title(r'Hitogramas')
        axs[1].plot(modificada[:,i], color = col, label = col)
        axs[1].set_xlabel(r'Valor pixel')
        axs[1].set_ylabel(r'Frecuencia Pixel')
        #axs[1].set_title(r'Hitograma')
    plt.show()
    

def lineal(img,contador):
    aux = len(img.shape)

    ##deficnion de los canales del siste
    ## Scar max y min del sistema
    min = img.min()
    max= img.max()
    alpha = 255/(max-min)
    beta= 50

    if aux ==3:
        b, g, r = cv2.split(img)
        ## Creacion de las matrices vacias de la funcion
        B = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        G = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        R = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)


        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                B[i,j] = linear_tranfor(b[i,j], alpha, beta)
                G[i,j] = linear_tranfor(g[i,j], alpha, beta)
                R[i,j] = linear_tranfor(r[i,j], alpha, beta)
        new = cv2.merge([B, G, R])

    else:
        new = np.zeros((img.shape[0], img.shape[1]),dtype=np.float64)
        # Interaccion para acceder a cada pixel de la funcion
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                new[i,j] = linear_tranfor(img[i,j], alpha, beta)


    name = "Pregunta_5_{}.png".format(contador)
    
    new = np.array(new, dtype = np.uint8)
    ## Nombre de los archivos a ser guardados
    show(img, new)
    guardar(new, name)

    ## Nombre de los archivos a ser guardados
    histograma, color = cal_histograma(img)
    modificado, color = cal_histograma(new)
    grafica(histograma, modificado, color)
    return None
def exponencial(img, contador):
    maximo = 255 
    base = 1.02
    C = 255.0/(np.power(base, maximo)-1)
    salida = C*(np.power(base, img)-1)
    norm = cv2.normalize(salida, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U);
    name = "Pregunta_14_{}.png".format(contador)

    show(img, norm)
    guardar(norm, name)

    histograma, color = cal_histograma(img)
    modificado, color = cal_histograma(norm)
    grafica(histograma, modificado, color)

    return None


def log(img,contador):
    c =255/(np.log(1+np.max(img)))
    log_image = c*(np.log(img+1))
    log_image = np.array(log_image, dtype= np.uint8)

    name = "Pregunta_6_{}.png".format(contador)
    
    show(img, log_image)
    guardar(log_image, name)

    histograma, color = cal_histograma(img)
    modificado, color = cal_histograma(log_image)
    grafica(histograma, modificado, color)
    return None

def equalize(img,contador):
    ## Equalizacion del histograma
    equ = cv2.equalizeHist(img)

    ## seccion para vizualizr el resultado
    name = "Pregunta_7_{}.png".format(contador)
    show(img, equ)
    guardar(equ, name)

    histograma, color = cal_histograma(img)
    modificado, color = cal_histograma(equ)
    grafica(histograma, modificado, color)

    return None 

def suma(imgs):
    ## Obtener los valores a sumar del sistem
    a = np.array(imgs[0,:,:], dtype=np.float64)/255
    b = np.array(imgs[1,:,:], dtype=np.float64)/255
    c = np.zeros((imgs.shape[0], imgs.shape[1]),dtype=np.float64)
    c = (a+b)
    c = c/2

    ## seccion para vizualizr el resultado
    name = "Pregunta_8_{}.png".format(1)
    cv2.imshow('Modificada', c)
    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    Final = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    ## Suma usando la funcionn de opencv 
    img1 = imgs[0,:,:]
    img2 = imgs[1,:,:]
    dst = cv2.addWeighted(img1,0.5,img2,0.5,0)
    name2 = "Pregunta_8_opencv.png"
    cv2.imwrite(os.path.join(path_w,name),Final)
    cv2.imwrite(os.path.join(path_w,name2),dst)
    return None

def suma_ponderada(imgs,alpha=0.2):
    ## Obtener los valores a sumar del sistem
    a = np.array(imgs[0,:,:], dtype=np.float64)/255
    b = np.array(imgs[1,:,:], dtype=np.float64)/255
    c = np.zeros((imgs.shape[0], imgs.shape[1]),dtype=np.float64)
    c = (alpha*a+(1-alpha)*b)
    c = c/1

    ## seccion para vizualizr el resultado
    name = "Pregunta_9_{}.png".format(1)
    cv2.imshow('Modificada', c)
    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    Final = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imwrite(os.path.join(path_w,name),Final)
    return None
    
def resta(imgs):
    print("Resta Manual")
    ## Obtener los valores a sumar del sistem
    a = np.array(imgs[0,:,:], dtype=np.float64)/255
    b = np.array(imgs[1,:,:], dtype=np.float64)/255
    c = np.zeros((imgs.shape[0], imgs.shape[1]),dtype=np.float64)
    c = np.abs(a-b)

    ## seccion para vizualizr el resultado
    name = "Pregunta_10_{}.png".format(1)
    cv2.imshow('Modificada', c)
    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    Final = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imwrite(os.path.join(path_w,name),Final)
    return None

def resta_opencv(imgs):
    print("Resta Open cv")
    a = imgs[0,:,:]
    b = imgs[1,:,:]
    #c =cv2.subtract(a, b)
    c =cv2.absdiff(a, b)
    name = "Pregunta_11_{}.png".format(1)
    cv2.imshow('Modificada', c)
    cv2.imshow('a', a)
    cv2.imshow('b', b)
    cv2.waitKey(0)
    Final = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imwrite(os.path.join(path_w,name),Final)
    
def conversion(img, contador):
    tipo = [cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YCR_CB, cv2.COLOR_BGR2GRAY]

    #new =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #new =cv2.cvtColor(img, tipo[contador])

    #new =cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    new = bgr2cmy(img)


    name = "Pregunta_12_{}.png".format(contador)

    ## Mostrar las imagenes por pantalla
    show(img, new)
    guardar(new, name)
    
    return None
    
def bgr2cmy(img):
    bgr =img.astype(np.float)/255
    k = 1-np.max(bgr, axis = 2)
    C = (1-bgr[...,2]-k)/(1-k)
    M = (1-bgr[...,1]-k)/(1-k)
    Y = (1-bgr[...,0]-k)/(1-k)

    CMYK = (np.dstack((C,M,Y,k))*255).astype(np.uint8)
    return CMYK
def diff_histogram(img):
    A = img[4 ,:, :]
    B = img[5, :, :]
    histogram_a, color_a = cal_histograma(A)
    histogram_b, color_b = cal_histograma(B)
    norma_A =np.sqrt(np.sum(np.diag(histogram_a.T@histogram_a)))
    norma_B =np.sqrt(np.sum(np.diag(histogram_b.T@histogram_b)))
    grafica(histogram_a, histogram_b, color_a)
    print(norma_A, norma_B)

#def grafica_histograma(a, b):
#    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#    axs[0].hist(1, a)
#    axs[1].hist(1, b)
#    plt.show()

    
def visual(imgs):
    contador =0
    for img in imgs: 
        #inver(img, contador)
        #rango(img, contador)
        #gamma(img, contador)
        #basic_linear_transform(img, contador)
        #lineal(img, contador)
        #log(img, contador)
        #exponencial(img, contador)
        #equalize(img, contador) ## Esta solo funciona con imagenes en Gray
        #conversion(img, contador)
        

        contador =contador+1

def main():
    imgs = data(path_o, 1)
    visual(imgs)
    #exponencial(imgs[0,:,:], 1)
    suma(imgs)
    #suma_ponderada(imgs, 0.2)
    #resta(imgs)
    #resta_opencv(imgs)
    #diff_histogram(imgs)
    
    

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate the while statement")
        pass

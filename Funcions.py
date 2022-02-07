"""
Funcions utilitzades per la lectura i codificacio d'una imatge
Marçal Garcia
Terrassa, ESEIAAT
Maig del 2020
"""

import numpy as np
import scipy.fftpack as fft
import sklearn.metrics as skl_m
import matplotlib.image as mpimg

"""
-----------------------------------RGB-----------------------------------------
"""
def rgb_channels_mgb(img):
    r=img.copy()
    g=img.copy()
    b=img.copy()
    r[:,:,1]=0
    r[:,:,2]=0
    g[:,:,0]=0
    g[:,:,2]=0
    b[:,:,0]=0
    b[:,:,1]=0
    return r,g,b

"""
-----------------------------------GRAY-----------------------------------------
"""
def rgb2gray_mgb(imatge_rgb):
    rgb=imatge_rgb.copy()
    r=rgb[:,:,0]
    g=rgb[:,:,1]
    b=rgb[:,:,2]
    img_bw=(0.2126*r+0.7152*g+0.0722*b).astype(int)
    return img_bw

"""
-----------------------------------BINARY-----------------------------------------
"""
def binary_mgb(img,thres):
    bw=img.copy()
    for i in range(len(bw)):
        for j in range(len(bw[i])):
            if bw[i,j]>=thres:
                bw[i,j]=1
            else:
                bw[i,j]=0
    return bw

"""
-----------------------------------PSNR----------------------------------------
"""
def PSNR(original,comprimida):
    mse=skl_m.mean_squared_error(original, comprimida)
    if (mse==0):
        return print('No noise')
    max_pixel = 255.0
    psnr = 10*np.log10((max_pixel)**2/mse)
    return psnr

"""
----------------------------------DCT/IDCT-------------------------------------
"""
def dct2(bloc):
    return fft.dct(fft.dct(bloc.T,norm='ortho').T,norm='ortho')
def idct2(bloc):
    return fft.idct(fft.idct(bloc.T,norm='ortho').T,norm='ortho')    

"""
---------------------------------COD JPEG--------------------------------------
Codification from a color-JPEG image to a grayscale image.
Attributes:
    Image's name
    Quantification matrix
    Quantification step
"""
def Cod_Imatge_DCT(imatge,Q,k):
    img = mpimg.imread(imatge,0)
    img = rgb2gray_mgb(img).astype(int)
    imsize = img.shape
    alcada = imsize[1]
    amplada = imsize[0]
    img_dividida = []
    y = 0 #[0,0]
    for i in range(8,alcada+1,8): #Take the values in each 8-pixel step
        x = 0
        for j in range(8,amplada+1,8):
            img_dividida.append(dct2(img[y:i,x:j]))
            x=j
        y=i
                
    M=np.array([[i*k for i in fila] for fila in Q])
    for bloc in img_dividida:
        for i in range(8):
            for j in range(8):
                bloc[i,j]=np.round(bloc[i,j]/M[i,j]).astype(int)*Q[i,j] 

    img_idct=[]
    for bloc in img_dividida:
        img_idct.append(np.round(idct2(bloc)).astype(int)) #IDCT for each block
    
    fila=0
    img_final=[]
    for i in range(64,len(img_idct)+1,64): #Blocks per row
        img_final.append(np.hstack((img_idct[fila:i]))) #Join the matrixes horizontally
        fila=i
    img_final=np.vstack((img_final)) #Join vertically the previous result
    return img_final

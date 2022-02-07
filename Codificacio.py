"""
Codificacio d'una imatge utilitzant JPEG
Marçal Garcia
Terrassa, ESEIAAT
Maig del 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

exec(open('Funcions.py').read()) #Run some functions

#Quantification matrix
Q=np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92]
            ,[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

"""
1. We read the image and we convert it into grayscale
"""
img = mpimg.imread("lena.png",0)
plt.figure()
plt.imshow(img)
plt.title('Original color image')
#print (img)
img = rgb2gray_mgb(img) #Conversion done: Y = 0.2126R + 0.7152G + 0.0722B

"""
2. Codification using DCT
"""
"ONE-BLOCK DCT"
posicio = 200
x=img[posicio:posicio+8,posicio:posicio+8]
y=dct2(x) #DCT
plt.figure()
plt.imshow(y,cmap='gray')
plt.title('DCT Coeficients')

"QUANTIFICATION"
k=10 #Compression's factor
M=[[i*k for i in fila] for fila in Q] 
I_q=np.round(y/M).astype(int) #One-block quantification
I_qq=I_q*Q

"ONE-BLOCK IDCT"
y_q=idct2(I_qq) #One-block IDCT

"PSNR CALCULATION"
psnr_bloc = PSNR(x,y_q)

"""
3. CODIFICATION OF ALL IMAGE
"""
lena_codificada=Cod_Imatge_DCT('lena.png',Q,k)
plt.imsave('lena_codificada_k{k}.jpeg'.format(k=k),lena_codificada,cmap='gray')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original image')
plt.subplot(1,2,2)
plt.imshow(lena_codificada,cmap='gray')
plt.title('Encoded image')

"PSNR AND ERROR CALCULATION"
psnr_imatge = PSNR(img,lena_codificada)
error = img-lena_codificada #Error
plt.figure()
plt.imshow(error,cmap='gray')
plt.title('Encoding error')


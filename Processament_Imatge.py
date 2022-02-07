"""
Processament de la imatge
Marçal Garcia
Terrassa, ESEIAAT
Maig del 2020
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

exec(open('Funcions.py').read())

img = mpimg.imread("lena.png",0)

"""
1. RGB channels
"""
r,g,b = rgb_channels_mgb(img)
plt.figure()
plt.subplot(1,3,1)
plt.title('R')
plt.imshow(r)
plt.subplot(1,3,2)
plt.title('G')
plt.imshow(g)
plt.subplot(1,3,3)
plt.title('B')
plt.imshow(b)


"""
2. Pass the image to Y
"""
img_y = rgb2gray_mgb(img)
plt.figure()
plt.imshow(img_y,cmap='gray')
plt.title('Grayscale image')

"""
3. Binary image
"""
threshold = 128
img_bw = binary_mgb(img_y,threshold)*255
plt.figure()
plt.imshow(img_bw,cmap='gray')
plt.title('Binary image with threshold={threshold}'.format(threshold=threshold))

#plt.hist(img,bins=60,alpha=1,edgecolor='black',linewidth=1)
#plt.hist(img_bw,bins=60,alpha=1,edgecolor='black',linewidth=1)




        

    
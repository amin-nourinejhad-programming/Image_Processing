import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
cwd = os.getcwd()
 
my_image = "shahgolii.jpg"

image=cv2.imread(my_image)
dic = {0:'blue',1:'green',2:'red'}
hist=cv2.calcHist([image],[0],None,[256],[0,256])


intensity_values = np.array([x for x in range(hist.shape[0])])


for i,j in dic.items():
    histr=cv2.calcHist([image],[i],None,[256],[0,256])
    
    plt.plot(intensity_values,histr,color=j,label= j+"channel")
    plt.xlim([0,256])

plt.show()

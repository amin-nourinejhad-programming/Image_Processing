import cv2
import matplotlib.pyplot as plt
import numpy as np

#defining the image 
my_image = "MRIbrain.jpg"
image = cv2.imread(my_image, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(15,15))

#plotting original image
plt.subplot(2,2,1)
plt.imshow(image,cmap="gray")
plt.title('Original')
hist = cv2.calcHist([image],[0],None,[256],[0 ,256])
intensity_values = np.array([x for x in range(hist.shape[0])])

# probability Mass Function 
PMF_hist = hist / (image.shape[0] * image.shape[1])
plt.subplot(2,2,3)
plt.bar(intensity_values,PMF_hist[:,0],color='red',width = 3)
plt.title('histogram of original image')
plt.xlabel('intensity values')
plt.ylabel('normalized occurance')

# constructing Negative image
Neg_image = -image +255 
hist1 = cv2.calcHist([Neg_image],[0],None,[256],[0 ,256])
PMF_hist1 = hist1 / (Neg_image.shape[0] * Neg_image.shape[1])
intensity_values1 = np.array([x for x in range(hist1.shape[0])])

#plotting negative image
plt.subplot(2,2,2)
plt.imshow(Neg_image,cmap="gray")
plt.title('Image Negatives')

# plotting histogram of negative image
plt.subplot(2,2,4)
plt.bar(intensity_values1,PMF_hist1[:,0],color='green',width = 3)
plt.title('histogram of Neg_image')
plt.xlabel('intensity values')
plt.ylabel('normalized occurance')

plt.tight_layout(pad =5.0)
plt.show()
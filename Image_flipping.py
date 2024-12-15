import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

dic={2:Image.FLIP_TOP_BOTTOM , 3: Image.FLIP_LEFT_RIGHT}
image=Image.open("cat.png")
#you can download the picture of cat from the address below:
#https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cat.png
plt.subplot(131)
plt.imshow(image)
for i,j in dic.items():
    plt.subplot(1,3,i)
    im = image.transpose(j)
    plt.imshow(im)
plt.tight_layout()
plt.show()

# there are a bunch of flipping images in terms of rotation as well:
# dic = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
#         "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
#         "ROTATE_90": Image.ROTATE_90,
#         "ROTATE_180": Image.ROTATE_180,
#         "ROTATE_270": Image.ROTATE_270,
#         "TRANSPOSE": Image.TRANSPOSE, 
#         "TRANSVERSE": Image.TRANSVERSE}
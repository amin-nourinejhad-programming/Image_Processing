import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

dic={2:Image.FLIP_TOP_BOTTOM , 3: Image.FLIP_LEFT_RIGHT}
image=Image.open("cat.png")
plt.subplot(131)
plt.imshow(image)
for i,j in dic.items():
    plt.subplot(1,3,i)
    im = image.transpose(j)
    plt.imshow(im)
plt.tight_layout()
plt.show()
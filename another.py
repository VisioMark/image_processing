from skimage import io
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("./images/download.jpg")
print(img.shape)
plt.imshow(img)
plt.show()

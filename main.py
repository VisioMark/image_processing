import cv2
import numpy as np
import utils

width= 1162
height = 1600


img = cv2.imread("images/001.png")

img = cv2.resize(img, (width, height)) # RESIZE IMAGE
imgContours = img.copy()
imgBlank = np.zeros((height,width, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY 


cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS


imgBlank = np.zeros_like(img) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
imgArray = ([img, imgGray, imgBlur, imgCanny], 
            [imgContours, imgBlank, imgBlank, imgBlank])
imgStacked = utils.stackImages(imgArray,0.5)
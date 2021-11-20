import cv2
import numpy as np
import imutils  # pip install imutils
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

img = cv2.imread('A_blue_eye.jpg')


# image en gris
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)


# filtre du noir
_, threshold = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY_INV) 	
canny_edges = cv2.Canny(threshold, 30, 200)
canny_iris = cv2.Canny(gray_img, 0, 50)


# filtre du blanc
_, threshold_white = cv2.threshold(gray_img, 75,255, cv2.THRESH_BINARY_INV) 	
canny_edges = cv2.Canny(threshold, 30, 200)

# coutour de la pupille
contours, hierarchy = cv2.findContours(
    canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# centre du cercle
c = contours[0]
M = cv2.moments(c)
x = int(M["m10"] / M["m00"])
y = int(M["m01"] / M["m00"])


# radius du cercle
cX,cY,w,h = cv2.boundingRect(contours[0]) 
radius = w / 2

"""# find white in image 
b,g,r = (img[x, y])
new_x = x
new_Y = y

while r < 200 or g < 200 or b < 200:
    b,g,r = (img[new_x, new_Y])
    #print("{} / {} / {}".format(r, g, b))
    #print("old : ", x)
    #print("new : ", new_x)
    cv2.circle(img, (new_x, y), 2, (255, 255, 255), -1)
    new_x = new_x - 1
"""
# draw the contour and center of the shape on the image
#cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
cv2.circle(img, (x, y), int(w / 2), (255, 255, 255), -1)
cv2.circle(img, (x, y), int(w * 2), (255, 255, 255), 1)

"""cv2.putText(img, "center", (x - 20, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)"""



# draw mask pupil 
mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
mask = cv2.circle(mask, (x, y), int(w / 2), (255, 255, 255), -1)
result = cv2.bitwise_not(img, img, mask = mask)
result[mask==255] = 255 # Color background white


# draw mask ext iris
mask = np.zeros(img.shape, dtype=np.uint8)
mask = cv2.circle(mask, (x, y), int(w * 2), (255,255,255), -1)
result = cv2.bitwise_and(img, mask)
result[mask==0] = 255 # Color background white


# show the image
cv2.imshow("Image", img)
cv2.imshow("result", result)
cv2.waitKey(0)


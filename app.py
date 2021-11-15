import cv2
import numpy

roi = cv2.imread('A_blue_eye.jpg')


# image en gris
gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)


# filtre du noir
_, threshold = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY_INV) 


# coutour de la pupille 
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(roi, contours[0] , -1, (0,0 , 255), 3)

cv2.imshow("gray", roi)
#cv2.imshow("gray treshold", threshold)

# waiting for key event
cv2.waitKey(0)
# destroying all windows
cv2.destroyAllWindows()



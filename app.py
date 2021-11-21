
import cv2
import numpy as np
import imutils  # pip install imutils
import sys


def verificationCircle(contours):
    for cercle in contours:
        # centre du cercle
        cX, cY, w, h = cv2.boundingRect(cercle)
        radius = w / 2
        print(radius)
        if radius > 10:
            return cercle

def isolate_iris(img):

    # image en gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # filtre du noir
    _, threshold = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY_INV)
    canny_edges = cv2.Canny(threshold, 30, 200)
    canny_iris = cv2.Canny(gray_img, 0, 50)

    # coutour de la pupille
    contours, hierarchy = cv2.findContours(
        canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = verificationCircle(contours)

    # centre du cercle
    M = cv2.moments(c)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # radius du cercle
    cX, cY, w, h = cv2.boundingRect(c)
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
    cv2.circle(img, (x, y), int(w / 2), (255, 255, 255), 1)
    cv2.circle(img, (x, y), int(w), (255, 255, 255), 1)

    """cv2.putText(img, "center", (x - 20, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)"""

    # draw mask pupil
    mask = np.full((img.shape[0], img.shape[1]), 0,
                   dtype=np.uint8)  # mask is only
    mask = cv2.circle(mask, (x, y), int(w / 2), (255, 255, 255), -1)
    result = cv2.bitwise_not(img, img, mask=mask)
    result[mask == 255] = 255  # Color background white

    # draw mask ext iris
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, (x, y), int(w), (255, 255, 255), -1)
    result = cv2.bitwise_and(img, mask)
    result[mask == 0] = 255  # Color background white

    return gray_img, threshold, canny_edges, result, x, y, radius




def polar_to_cartesian (image, x, y, radius):
    # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
    polar_img = cv2.warpPolar(image, (256, 1024), (x, y), radius * 2, cv2.WARP_POLAR_LINEAR)
    # Rotate it sideways to be more visually pleasing
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_img = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(polar_img, 100, 255, cv2.THRESH_BINARY_INV)
    return polar_img, threshold

if __name__ == '__main__':

    img = cv2.imread('test4.jpg')
    gray_img, threshold, canny_edges, result, x, y, radius = isolate_iris(img)

    polar_img, polar_threshold = polar_to_cartesian(result, x, y, radius)

    # show the image
    cv2.imshow("Image", img)
    #cv2.imshow("gray", gray_img)
    #cv2.imshow("thres", threshold)
    #cv2.imshow("canny", canny_edges)
    cv2.imshow("result", result)
    cv2.imshow("cart", polar_img)
    cv2.imshow("pt", polar_threshold)
    cv2.waitKey(0)

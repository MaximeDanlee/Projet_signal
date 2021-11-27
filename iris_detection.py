import cv2
import numpy as np
import os
import math
from math import hypot

gamma = -48


# gamma is -48 for UBIRIS database

def polar_to_cartesian (image, center, radius):

    # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
    polar_img = cv2.warpPolar(image, (256, 1024), center, radius * 2, cv2.WARP_POLAR_LINEAR)
    # Rotate it sideways to be more visually pleasing
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # crop image 
    polar_img = polar_img[int(polar_img.shape[0] / 2) : polar_img.shape[0], 0 : polar_img.shape[1]]

    _, threshold = cv2.threshold(polar_img, 100, 255, cv2.THRESH_BINARY_INV)
    return polar_img, threshold

def Grabcut(image):  # function to differentiat foreground and background
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h = image.shape[0]
    w = image.shape[1]
    rect = (int(w/4), int(h/4), int(w*3/4), int(h*3/4))
    """
    cv2.rectangle(image,(int(w/4),int(h/4)),(int(w*3/4),int(h*3/4)),(0,255,0),2)
    #cv2.imshow("rect",image)
    #cv2.waitKey(0)"""
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image * mask2[:, :, np.newaxis]
    return img


def noise_reduction(image):  # all noise cancelling process for CHT
    inverted_gray = cv2.bitwise_not(image)
    kernel = np.ones((5, 5), np.uint8)
    black_hat = cv2.morphologyEx(inverted_gray, cv2.MORPH_BLACKHAT, kernel)
    #cv2.imshow("Black_hat" + name, black_hat)
    #cv2.waitKey(0)
    no_reflec = cv2.add(inverted_gray, black_hat)
    median_blur = cv2.medianBlur(no_reflec, 5)
    cv2.equalizeHist(median_blur)
    #cv2.imshow("Median_blur " + name, median_blur)
    #cv2.waitKey(0)

    retval, thres_image = cv2.threshold(cv2.bitwise_not(median_blur), 100, 255, cv2.THRESH_BINARY_INV)

    #cv2.imshow("thresh " + name, thres_image)
    #cv2.waitKey(0)
    canny = cv2.Canny(thres_image, 200, 100)
    return canny


images = []
names = []
# running the complete set of image database from the folder ”data" for filename in

for filename in os.listdir("data"):
    if filename is not None:
        image = cv2.imread(os.path.join("data", filename), 1)
        images.append(image)
        names.append(filename.split('.')[0])

for i in range(len(images)):
    name = names[i]
    image = images[i]

    ##cv2.imshow("input_" + name, image)
    #cv2.waitKey(0)
    x, y, z = image.shape


    # crop only eye
    #image = Grabcut(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow("Grab_cut " + name, gray_image)
    #cv2.waitKey(0)


    # circle detection 
    ipfilter = noise_reduction(gray_image)
    ##cv2.imshow("CED " + name, ipfilter)
    #cv2.waitKey(0)
    circles = cv2.HoughCircles(ipfilter, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20, minRadius=0)

    if circles is not None:
        inner_circle = np.uint16(np.around(circles[0][0])).tolist()
    cv2.circle(image, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 1)
    ##cv2.imshow("HoughCircle " + name, image)
    #cv2.waitKey(0)


    # crop image by comparing each pixel dsitance from center with radius
    for j in range(x):
        for k in range(y):
            if hypot(k - inner_circle[0], j - inner_circle[1]) >= inner_circle[2]:
                gray_image[j, k] = 0
    ##cv2.imshow("output2_" + name, gray_image)

    # polar to cartesian
    polar_img, polar_threshold = polar_to_cartesian (gray_image, (inner_circle[0], inner_circle[1]), inner_circle[2])
    ##cv2.imshow("polar_img", polar_img)
    ##cv2.imshow("polar_threshold ", polar_threshold )
    #cv2.waitKey(0)

    cv2.imwrite(f'result/result{i}.png', polar_img) 
    print(f"done : {i}")
    
    #cv2.destroyAllWindows()

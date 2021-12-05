import cv2
import numpy as np
import os
import math
from math import hypot
from compare import compare


def polar_to_cartesian (image, center, radius):

    # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
    polar_img = cv2.warpPolar(image, (256, 1024), center, radius * 2, cv2.WARP_POLAR_LINEAR)
    # Rotate it sideways to be more visually pleasing
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # crop image 
    polar_img = polar_img[int(polar_img.shape[0] / 2) : polar_img.shape[0], 0 : polar_img.shape[1]]

    _, threshold = cv2.threshold(polar_img, 100, 255, cv2.THRESH_BINARY_INV)
    return polar_img, threshold

def grab_cut(image):  # fonction qui permet de différencier le premier du second plan : isoler second plan
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


def noise_reduction(image):
    inverted_gray = cv2.bitwise_not(image)                                      # inversion du gris
    kernel = np.ones((5, 5), np.uint8)                                          # création d'un tableau de 5/5
    black_hat = cv2.morphologyEx(inverted_gray, cv2.MORPH_BLACKHAT, kernel)     # Ici, dans cette image, tous les objets qui sont blancs sur un fond sombre sont mis en évidence en raison de la transformation Black Hat appliquée à l'image d'entrée.
    no_reflec = cv2.add(inverted_gray, black_hat)                               # on additionne l'image grise et l'image black_hat
    median_blur = cv2.medianBlur(no_reflec, 5)                                  # on enlève le bruit 
    
    #cv2.imshow("Black_hat" + name, black_hat)
    #cv2.imshow("inverted_gray" + name, inverted_gray)
    #cv2.imshow("no_reflec" + name, no_reflec)
    #cv2.imshow("Median_blur" + name, median_blur)
    #cv2.waitKey(0)

    retval, thres_image = cv2.threshold(cv2.bitwise_not(median_blur), 100, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("thresh " + name, thres_image)
    #cv2.waitKey(0)
    canny = cv2.Canny(thres_image, 200, 100)                                    # on récupère les contours de l'image
    return canny


images = []
names = []
# parcourir toutes les images dans le dossier data
for filename in os.listdir("data"):
    if filename is not None:
        image = cv2.imread(os.path.join("data", filename), 1)
        images.append(image)
        names.append(filename.split('.')[0])

for i in range(len(images)):
    name = names[i]
    image = images[i]
    x, y, z = image.shape


    # Isoler l'oeil pour avoir moins d'information à traiter
    # image = Grabcut(image)


    # Transformer l'image en gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow("Grab_cut " + name, gray_image)
    #cv2.waitKey(0)


    # filtre de canny
    ipfilter = noise_reduction(gray_image)
    cv2.imshow("CED " + name, ipfilter)
    cv2.waitKey(0)

    # repérer les cercles dans l'image
    circles = cv2.HoughCircles(ipfilter, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20, minRadius=0)

    if circles is not None:
        inner_circle = np.uint16(np.around(circles[0][0])).tolist()
    cv2.circle(image, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 1)
    cv2.imshow("HoughCircle " + name, image)
    cv2.waitKey(0)


    # recadrer l'image grâce au centre et au rayon du cercle
    mask = np.zeros(gray_image.shape,np.uint8)
    new_image = cv2.circle(mask, (inner_circle[0], inner_circle[1]), inner_circle[2], (255, 255, 255), -1)
    gray_image = cv2.bitwise_and(gray_image,gray_image,mask=mask)

    cv2.imshow("output2_" + name, gray_image)
    cv2.waitKey(0)

    # polair vers cartesian
    cart_img, cart_threshold = polar_to_cartesian(gray_image, (inner_circle[0], inner_circle[1]), inner_circle[2])
    cv2.imshow("Cartesian" + name, cart_img)
    cv2.imshow("Cartesian_thresh" + name, cart_threshold )
    cv2.waitKey(0)


    # sauvegarder le resultat sur le disque
    cv2.imwrite(f'result/result{i}.png', cart_threshold) 
    
    # compare result
    compare()
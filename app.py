
import cv2
import numpy as np

def verificationCircle(contours):
    for cercle in contours:
        # centre du cercle
        cX, cY, w, h = cv2.boundingRect(cercle)
        radius = w / 2
        print(radius)
        if radius > 10:
            return cercle

def Grabcut(image):  # function to differentiat foreground and background
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h = image.shape[0]
    w = image.shape[1]
    rect = (int(w/4), int(h/4), int(w*3/4), int(h*3/4))
    """
    cv2.rectangle(image,(int(w/4),int(h/4)),(int(w*3/4),int(h*3/4)),(0,255,0),2)
    cv2.imshow("rect",image)
    cv2.waitKey(0)"""
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image * mask2[:, :, np.newaxis]
    return img


def noise_reduction(image):  # all noise cancelling process for CHT
    inverted_gray = cv2.bitwise_not(image)
    cv2.imshow("inverted_gray", inverted_gray)
    cv2.waitKey(0)
    kernel = np.ones((5, 5), np.uint8)
    black_hat = cv2.morphologyEx(inverted_gray, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("Black_hat", black_hat)
    cv2.waitKey(0)
    no_reflec = cv2.add(inverted_gray, black_hat)
    median_blur = cv2.medianBlur(no_reflec, 5)
    #cv2.equalizeHist(median_blur)
    cv2.imshow("Median_blur ", median_blur)
    cv2.waitKey(0)

    retval, thres_image = cv2.threshold(cv2.bitwise_not(median_blur), 100, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("thresh ", thres_image)
    cv2.waitKey(0)
    canny = cv2.Canny(thres_image, 200, 100)
    return canny


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

    img = cv2.imread('test.jpg')

    img = Grabcut(img) # isoler l'oeil

    img_canny = noise_reduction(img)

    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20, minRadius=0)

    if circles is not None:
        inner_circle = np.uint16(np.around(circles[0][0])).tolist()
    cv2.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 1)
    cv2.imshow("HoughCircle ", img)
    cv2.waitKey(0)

    #gray_img, threshold, canny_edges, result, x, y, radius = isolate_iris(img)

    #polar_img, polar_threshold = polar_to_cartesian(result, x, y, radius)

    # show the image
    cv2.imshow("Image", img)
    #cv2.imshow("gray", gray_img)
    #cv2.imshow("thres", threshold)
    #cv2.imshow("canny", canny_edges)
    #cv2.imshow("result", result)
    #cv2.imshow("cart", polar_img)
    #cv2.imshow("pt", polar_threshold)
    cv2.waitKey(0)

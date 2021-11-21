circles = cv2.HoughCircles(canny_edges, cv2.HOUGH_GRADIENT, 2, img.shape[0]/2)

if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circlesRound = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circlesRound:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    cv2.imwrite(filename = 'test.circleDraw.png', img = img)
    cv2.imwrite(filename = 'test.circleDrawGray.png', img = canny_edges)
else:
    print ('No circles found')
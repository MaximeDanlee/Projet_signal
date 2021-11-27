import cv2
import numpy as np
import os

def compare():
    #--- take the absolute difference of the images ---
    imagetoCompare = []
    images = []
    names = []
    results = []

    # find image to compare
    for person in os.listdir("result"):
        if person is not None:
            image = cv2.imread(os.path.join(f"result/", person), 1)
            imagetoCompare = image

    # find all images in database
    for person in os.listdir("database"):
        for filename in os.listdir(f"database/{person}"):
            if filename is not None:
                image = cv2.imread(os.path.join(f"database/{person}", filename), 1)
                images.append(image)
                names.append(person)


    for i in range(len(images)):
        name = names[i]
        image = images[i]

        res = cv2.absdiff(imagetoCompare, image)

        #--- convert the result to integer type ---
        res = res.astype(np.uint8)

        #--- find percentage difference based on number of pixels that are not zero ---
        percentage = (np.count_nonzero(res) * 100)/ res.size
        results.append([name, percentage])

    score = 1000
    nameResult = "Nobody"
    for result in results:
        if result[1] < score and result[1] < 20:
            score = result[1]
            nameResult = result[0]

    if score != 1000:
        print("c'est", nameResult, "à", round(100 - score, 2), "%")
    else:
        print("Nobody")
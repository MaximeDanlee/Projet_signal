import cv2
import numpy as np
import os

def compare():
    imagetoCompare = []
    images = []
    names = []
    results = []

    # récupère l'image à comparer
    for person in os.listdir("result"):
        if person is not None:
            image = cv2.imread(os.path.join(f"result/", person), 1)
            imagetoCompare = image

    # récupère les images dans la base de données
    for person in os.listdir("database"):
        for filename in os.listdir(f"database/{person}"):
            if filename is not None:
                image = cv2.imread(os.path.join(f"database/{person}", filename), 1)
                images.append(image)
                names.append(person)


    for i in range(len(images)):
        name = names[i]
        image = images[i]

        res = cv2.absdiff(imagetoCompare, image)                    # permet de faire la différence entre les deux images

        res = res.astype(np.uint8)                                  # converti le resultat en entier

        percentage = (np.count_nonzero(res) * 100)/ res.size        # trouver la différence de pourcentage en fonction du nombre de pixels qui ne sont pas nuls
        results.append([name, percentage])

    score = 1000
    nameResult = "Nobody"
    for result in results:
        if result[1] < score and result[1] < 20:                    # permet de trouver l'image qui à le plus grand pourcentage et à minimum un pourcentage de 80%
            score = result[1]
            nameResult = result[0]

    if score != 1000:
        print("c'est", nameResult, "à", round(100 - score, 2), "%")
    else:
        print("Nobody")
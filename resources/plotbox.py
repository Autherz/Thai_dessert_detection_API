import cv2
import os


def plotBoxToImage(dataset, image_pat, image_path):

    img = cv2.imread(image_pat)
    i = 0
    dataset = dataset[0]
    for data in dataset:
        print(i, data[2], data[4], data[3], data[5])
        i += 1
        x = min(int(data[2]), int(data[4]))
        y = min(int(data[3]), int(data[5]))
        w = abs(int(data[2])-int(data[4]))
        h = abs(int(data[3])-int(data[5]))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite(os.path.join(image_path), img)
    
        


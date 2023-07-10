import csv
import cv2

import ImageProcessing as ip


class Database:

    def __init__(self, file, images_path):
        self.dictionary = load(file)
        self.images_path = images_path
        self.keypoints = None

    def getItem(self, index):
        return self.dictionary[index]

    def getTitle(self, index):
        return self.getItem(index)["Title"]

    def getAuthor(self, index):
        return self.getItem(index)["Author"]

    def getRoom(self, index):
        return self.getItem(index)["Room"]

    def getImageName(self, index):
        return self.getItem(index)["Image"]

    def getImage(self, index):
        return cv2.imread(self.images_path + "/" + self.getImageName(index), cv2.IMREAD_COLOR)

    def computeKeypoints(self, factor):
        orb = cv2.ORB_create()
        self.keypoints = []
        for i in range(len(self.dictionary)):
            image = self.getImage(i)
            image = ip.resize(image, factor)
            kp_image, des_image = orb.detectAndCompute(image, None)
            self.keypoints.append([kp_image, des_image])

    def getKeypoints(self, index):
        if self.keypoints is None:
            return None
        else:
            return self.keypoints[index]

    def length(self):
        return len(self.dictionary)


# Loader of the db file
def load(file):
    reader = csv.DictReader(open(file, 'r'))
    dictionary = []
    for line in reader:
        dictionary.append(line)
    return dictionary

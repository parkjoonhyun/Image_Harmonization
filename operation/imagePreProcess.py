from os import listdir
from os.path import isfile , join

import torch
from PIL import Image , ImageOps
from random import randrange
import numpy as np
import glob
import cv2
from data.dataLoader import setDataLoader
from torchvision import transforms
from model.modelUtils import StyleEncoder

def randomCrop(image , targetSize ):
    imgSize = image.size
    xMax = imgSize[0] - targetSize
    yMax = imgSize[1] - targetSize
    randomCroppedIMG = []

    for i in range(2):
        random_x = randrange(0, xMax//2 + 1) * 2
        random_y = randrange(0, yMax//2 + 1) * 2
        area = (random_x, random_y, random_x + targetSize, random_y + targetSize)
        c_img = image.crop(area)
        fit_img_h = ImageOps.fit(c_img, (224, 224), Image.ANTIALIAS)
        randomCroppedIMG.append(fit_img_h)
        # fit_img_h.save(path+'RandomCrop__'+str(i)+file)
    # randomCroppedIMG = torch.stack(randomCroppedIMG[0],randomCroppedIMG[1])
    return randomCroppedIMG

def convertLUT(imageData):
    LUTpaths = glob.glob("C:/workspace/SSIH/dataset/LUT_pack/*.png")
    path = np.random.choice(LUTpaths, 1, replace=False)
    LUTImage = cv2.imread(path[0])[:, :, ::-1]
    LUTImages = []
    for data in imageData:
        data = np.array(data)
        newImage = data.copy()
        for y in range(224):
            for x in range(224):
                r, g, b = data[y, x]
                r = r // 4
                g = g // 4
                b = b // 4
                bh = (b // 8) * 64
                bw = (b % 8) * 64
                r, g, b = LUTImage[bh + g, bw + r]
                newImage[y, x] = [r, g, b]
        newImage = Image.fromarray(newImage)
        data = Image.fromarray(data)
        LUTImages.append(newImage)
    return LUTImages

def PILtoTensor(imageData):
    imageData = np.array(imageData)
    pilToTensor = transforms.ToTensor()
    temp = []
    for data in imageData:
        tensor = pilToTensor(data)
        temp.append(tensor)
    output = torch.stack(temp[0:len(imageData)])
    return output

def Tensor2PIL(tensorData):
    trans = transforms.ToPILImage()
    imageData = trans(tensorData)
    return imageData

# for i , x in enumerate(dataLoader):
#     for data in x[0]:
#         imageData = Tensor2PIL(data)
#         randomCroppedIMG = randomCrop(imageData , 224 )
#         randomCroppedIMG[0].show()
#         randomCroppedIMG[1].show()
#         print(len(randomCroppedIMG))


def Preprocess(tensorData):
    contentA =[]
    contentB =[]
    referenceA =[]
    referenceB =[]
    for data in tensorData[0]:
        imageData = Tensor2PIL(data)
        randomCroppedIMG = randomCrop(imageData, 224)
        LUTImages = convertLUT(randomCroppedIMG)
        contentA.append(randomCroppedIMG[0])
        referenceA.append(randomCroppedIMG[1])
        contentB.append(LUTImages[0])
        referenceB.append(LUTImages[1])

    contentA = PILtoTensor(contentA)
    referenceA = PILtoTensor(referenceA)
    contentB = PILtoTensor(contentB)
    referenceB = PILtoTensor(referenceB)
    return contentA , referenceA , contentB , referenceB






# class Preprocess():
#     def __init__(self):
#         self.randomCrop() = randomCrop()
#
#     def randomOverlapCrop(path):
#         root_path = path + '\\'
#         files = []
#         for i in listdir(root_path):
#             if isfile(join(root_path, i)):
#                 files.append(i)
#
#         randomCroppedImage = []
#         for file in files:
#             path = root_path + file
#             img = Image.open(path, 'r')
#             img = img.resize((300,300))
#             randomCroppedImage.append(randomCrop(img , 224 , root_path , file))
#
#         return randomCroppedImage




from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import datetime
import pickle
from os import listdir
from PIL import Image , ImageEnhance
from os.path import isfile, join

def setDataLoader(path):
    dataTransforms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406) , (0.229 , 0.224, 0.225)),
    ])
    inputDS = datasets.ImageFolder(path , transform = dataTransforms)
    dataLoader = DataLoader(inputDS, batch_size=8, num_workers=0, shuffle=False)
    return dataLoader

# dataLoader = setDataLoader("C:\workspace\SSIH\dataset")
#
# for i , x in enumerate(dataLoader):
#     x = x[0]
#
#     for i,data in enumerate(x):
#         trans = transforms.ToPILImage()
#         imageData = trans(data)
#         imageData.show()
#         numpyData = np.array(imageData)
#         trams = transforms.ToTensor()
#         tensorData = trams(numpyData)
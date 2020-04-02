from ImageServer import ImageServer
import numpy as np

imageDirs = ["../data/images/Manga_Cropped_60_Points/test/"]
datasetDir = "../data/"

meanShape = np.load("../data/mangaMeanFaceShape.npz")["meanShape"]

mangaSet = ImageServer(initialization='rect')
mangaSet.PrepareData(imageDirs, None, meanShape, 0, 144, False)
mangaSet.LoadImages()
mangaSet.CropResizeRotateAll()
mangaSet.imgs = mangaSet.imgs.astype(np.float32)
mangaSet.Save(datasetDir, "mangaSet.npz")

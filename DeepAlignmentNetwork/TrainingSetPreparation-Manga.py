from ImageServer import ImageServer
import numpy as np

trainDirs = ["../data/images/Manga_Cropped_60_Points_v2/train/"]
validationDirs = ["../data/images/Manga_Cropped_60_Points_v2/validation/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]

datasetDir = "../data/GeoAug-10x-0.5-0415/"

meanShape = np.load("../data/mangaMeanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(trainDirs, None, meanShape, 0, 1151, True)
trainSet.LoadImages()

# for aumentation
trainSet.GeneratePerturbations(10, [0.1, 0.1, 20, 0.1])

# geometric augmentation
trainSet.GeometricAugmentation(p_geom=0.5)

# for no augmentation
#trainSet.CropResizeRotateAll()
#trainSet.imgs = trainSet.imgs.astype(np.float32)

trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='rect')
validationSet.PrepareData(validationDirs, None, meanShape, 0, 144, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)

from scipy import ndimage
import numpy as np
import utils
try:
    import cPickle as pickle
except:
    import _pickle as pickle
import glob
from os import path
import matplotlib
import geoAug
import cv2
from skimage.morphology import thin
matplotlib.use('Agg')
import matplotlib.pyplot as plt  

class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box', color=False):
        self.origLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])

        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.color = color;

        self.boundingBoxes = []

    @staticmethod
    def Load(filename):
        imageServer = ImageServer()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)

        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, np.newaxis]

        return imageServer

    def Save(self, datasetDir, filename=None):
        if filename is None:
            filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        filenames = []
        landmarks = []
        boundingBoxes = []


        for i in range(len(imageDirs)):
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png")

            if boundingBoxFiles is not None:
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'))

            for j in range(len(filenamesInDir)):
                filenames.append(filenamesInDir[j])
                
                ptsFilename = filenamesInDir[j][:-3] + "pts"
                landmarks.append(utils.loadFromPts(ptsFilename))

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    boundingBoxes.append(boundingBoxDict[basename])
                

        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]

        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:     
            mirrorList = mirrorList + [True for i in range(nImgs)]
            filenames = np.concatenate((filenames, filenames))

            landmarks = np.vstack((landmarks, landmarks))
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))       

        self.origLandmarks = landmarks
        self.filenames = filenames
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes

    def LoadImages(self):
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []

        for i in range(len(self.filenames)):
            img = ndimage.imread(self.filenames[i])

            if self.color:
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                self.origLandmarks[i] = utils.mirrorShape(self.origLandmarks[i], img.shape)
                img = np.fliplr(img)

            if self.color:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis]

            groundTruth = self.origLandmarks[i]

            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape)
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])
            elif self.initialization == 'whole':
                box = np.array([0, 0, img.shape[2]-1, img.shape[1]-1])
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=box)

            self.imgs.append(img)
            self.initLandmarks.append(bestFit)
            self.gtLandmarks.append(groundTruth)

        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)    

    def GeneratePerturbations(self, nPerturbations, perturbations):
        self.perturbations = perturbations
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []           

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180         
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())
        print("Creating perturbations of " + str(self.gtLandmarks.shape[0]) + " shapes")

        for i in range(self.initLandmarks.shape[0]):
            print(i)
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy()

                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])     
            
                tempInit = tempInit + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)            
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])                

                newImgs.append(tempImg)
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)

    def CropResizeRotateAll(self):
        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []   
        transform = []
        
        for i in range(self.initLandmarks.shape[0]):
            tempImg, tempInit, tempGroundTruth, A, t = self.CropResizeRotate(self.imgs[i], self.initLandmarks[i], self.gtLandmarks[i])

            newImgs.append(tempImg)
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)
            transform.append({'A':A, 't':t})

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks) 
        self.transform = np.array(transform) 

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg
        
        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        
        self.imgs = self.imgs / self.stdDevImg


        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()  
        meanImg = meanImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(meanImg, (1, 2, 0)))
        else:
            plt.imshow(meanImg[0], cmap=plt.cm.gray)
        plt.savefig("../meanImg.jpg")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()  
        stdDevImg = stdDevImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
        else:
            plt.imshow(stdDevImg[0], cmap=plt.cm.gray)
        plt.savefig("../stdDevImg.jpg")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)

        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgSize[::-1]) / 2
        destShape += offset

        A, t = utils.bestFit(destShape, initShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth, A, t

    def GeometricAugmentation(self, p_geom=0.):
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            img = img[0].astype('uint8')
            img, self.gtLandmarks[i] = geoAug.augment_menpo_img_geom(img, self.gtLandmarks[i], p_geom)
            img = img[np.newaxis]
            self.imgs[i] = img
    
    '''
    def ToOriginalSize(self, index, pred_lms):
        A = self.transform[index][0]
        t = self.transform[index][1]
        A2 = np.linalg.inv(A)
        orig_size_lms = np.dot(pred_lms - t, A2)
        return self.originalImage[index], orig_size_lms
    '''

    def adaptiveThresholding(self, blockSize=21, param=50):
        for i in range(self.imgs.shape[0]):
            if self.color:
                img = np.mean(self.imgs[i], axis=0)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, param)
                img = np.array([img, img, img])
                self.imgs[i] = img
            else:
                img = self.imgs[i][0]
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, param)
                self.imgs[i] = img[np.newaxis]
    
    def inverse(self):
        for i in range(self.imgs.shape[0]):
            self.imgs[i] = 255 - self.imgs[i]

    def thin(self):
        for i in range(self.imgs.shape[0]):
            if self.color:
                img = np.mean(self.imgs[i], axis=0)
                img = np.where(img == 255, 0, 255)
                img = thin(img, 1)
                img - np.array(img, dtype='uint8')
                img = np.where(img < 1, 255, 0)
                img = np.array([img, img, img])
                self.imgs[i] = img
            else:
                img = self.imgs[i][0]
                img = np.where(img == 255, 0, 255)
                img = thin(img, 1)
                img - np.array(img, dtype='uint8')
                img = np.where(img < 1, 255, 0)
                self.imgs[i] = img[np.newaxis]
import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

datasetDir ="../data/"

verbose = True
showResults = True
showCED = True

normalization = 'mangaChin'
failureThreshold = 0.0333

networkFilename = "../networks/network_00150_2020-04-01-14-26.npz"
network = FaceAlignment(112, 112, 1, nStages=1)
network.loadNetwork(networkFilename)

print ("Network being tested: " + networkFilename)
print ("Normalization is set to: " + normalization)
print ("Failure threshold is set to: " + str(failureThreshold))

# commonSet = ImageServer.Load(datasetDir + "commonSet.npz")
# challengingSet = ImageServer.Load(datasetDir + "challengingSet.npz")
# w300 = ImageServer.Load(datasetDir + "w300Set.npz")
mangaSet = ImageServer.Load(datasetDir + "mangaSet.npz")

'''
print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
commonErrs = tests.LandmarkError(commonSet, network, normalization, showResults, verbose)
print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingErrs = tests.LandmarkError(challengingSet, network, normalization, showResults, verbose)

fullsetErrs = commonErrs + challengingErrs
print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
print("Average error: {0}".format(np.mean(fullsetErrs)))
tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)

print ("Processing 300W private test set")
w300Errs = tests.LandmarkError(w300, network, normalization, showResults, verbose)
tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)
'''

print ("Processing manga test set")
mangaErrs = tests.LandmarkError(mangaSet, network, normalization, showResults, verbose)
tests.AUCError(mangaErrs, failureThreshold, showCurve=showCED)

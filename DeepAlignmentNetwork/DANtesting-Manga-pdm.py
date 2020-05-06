import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests_pdm

datasetDir ="../data/"

verbose = True
showResults = True
showCED = True

normalization = 'mangaChin'
failureThreshold = 0.0333

networkFilename = "../networks/network151.npz"
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
#mangaErrs = tests_pdm.LandmarkError(mangaSet, network, normalization, showResults, verbose)
#tests_pdm.AUCError(mangaErrs, failureThreshold, showCurve=showCED)
'''
r_brow_pc_list = [2, 3, 4, 5, 6] # max 6
l_brow_pc_list = [2, 3, 4, 5, 6] # max 6
r_eye_pc_list = [2, 3, 4, 6, 8, 12, 16] # max 8 [2, 3, 4, 6, 8] -> 16
l_eye_pc_list = [2, 3, 4, 6, 8, 12, 16] # max 8 -> 16
mouth_pc_list = [2, 3, 4, 6, 8, 12, 16] # max 36 -> 16
chin_pc_list = [5, 7, 10, 16, 20] # max 30

all_pc_list = [30]

part_pc_lists = [
    r_brow_pc_list, l_brow_pc_list, r_eye_pc_list, l_eye_pc_list,
     mouth_pc_list, chin_pc_list]
parts = ['r_brow', 'l_brow', 'r_eye', 'l_eye', 'mouth', 'chin']
default_opt = [2, 2, 3, 3, 6, 7]
best_opt = [0, 0, 0, 0, 0, 0]
for i in range(len(part_pc_lists)):
    part_pc_list = part_pc_lists[i]
    lowestMeanEorrors = 100
    for j in range(len(part_pc_list)): 
        pc_opt = default_opt[:]
        pc_opt[i] = part_pc_list[j]
        errors = tests_pdm.LandmarkError(mangaSet, network, normalization, showResults=False, verbose=False, pdm=True, pc_opt=pc_opt)
        meanErrors = np.mean(errors)
        if meanErrors < lowestMeanEorrors:
            lowestMeanEorrors = meanErrors
            default_opt[i] = part_pc_list[j]
            print(default_opt)
            best_opt[i] = part_pc_list[j]
    print('Best pdm for' + parts[i] + ' -> PCs: ' + str(best_opt[i]) + ' Error: ' + str(lowestMeanEorrors))
print('Default opt: ', default_opt)
print('Best opt: ', best_opt)
'''
mangaErrs = tests_pdm.LandmarkError(mangaSet, network, normalization, showResults, verbose, pdm=True, pc_opt=[2, 2, 6, 8, 12, 10])
tests_pdm.AUCError(mangaErrs, failureThreshold, showCurve=showCED)

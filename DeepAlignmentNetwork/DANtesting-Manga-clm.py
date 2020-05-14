import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
from MangaPredictor import MangaPredictor
import tests_pdm
import utils
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
mangaSet = ImageServer.Load(datasetDir + "mangaSet_0513.npz")


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

pcs = [0]
mangaPredictor = MangaPredictor(network)
for pc in pcs:
    
    #print('Predict With Partial CLM.  principal components: ' + str(pc))
    #ectp_list = mangaPredictor.predictWithPartialCLM(mangaSet, clm_dir='../data/clm_models/', n_active_components=pc, self_targeting=False, opt=None)
    
    # print('Predict With Global CLM.  principal components: ' + str(pc))
    # ect_list = mangaPredictor.predictWithGlobalCLM(mangaSet, clm_path='../data/clm_models/manga_all', n_active_components=pc, self_targeting=True, opt=None)
    
    print('Predict With Pure DAN.')
    e_list = mangaPredictor.predictPureDAN(mangaSet)
    
    #print('Predict With Partial PDM.')
    #ecp_list = mangaPredictor.predictWithPartialPDM(mangaSet, pc_opt=[2, 2, 8, 6, 12, 10])

    # calculate error of each part
    left_brow_inds = np.arange(47, 52)
    right_brow_inds = np.arange(52, 57)
    left_eye_inds = np.arange(27, 37) # np.append(np.arange(27, 37), 57)
    right_eye_inds = np.arange(37, 47) # np.append(np.arange(37, 47), 58)
    mouth_inds = np.arange(17, 27)
    jaw_line_inds = np.arange(0, 17)

    parts = ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'mouth', 'chin']
    part_inds_opt = [left_brow_inds, right_brow_inds, left_eye_inds, right_eye_inds, mouth_inds, jaw_line_inds]
    
    for i, part in enumerate(parts):
        part_inds = part_inds_opt[i]
        print(part)
        mangaErrs = tests_pdm.LandmarkError(gt_lms_list=mangaSet.gtLandmarks, pred_lms_list=e_list, part_inds=part_inds, normalization='mangaChin', showResults=False, verbose=False)

    print('whole')
    mangaErrs = tests_pdm.LandmarkError(gt_lms_list=mangaSet.gtLandmarks, pred_lms_list=e_list, normalization='mangaChin', showResults=False, verbose=False)

    A = mangaSet.transform[0]['A']
    t = mangaSet.transform[0]['t']
    A2 = np.linalg.inv(A)
    orig_size_lms = np.dot(e_list[0] - t, A2)
    filename = mangaSet.filenames[0]
    print(filename)
    utils.saveToPts('abjasdf_purDAN.pts', orig_size_lms)
#mangaErrs = tests_pdm.LandmarkError(mangaSet, network, normalization, showResults, verbose, pdm=True, pc_opt=[2, 2, 6, 8, 12, 10])
#tests_pdm.AUCError(mangaErrs, failureThreshold, showCurve=showCED)

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path
from pdm_clm_functions_manga import feature_based_pdm_corr, clm_correct
from menpo.image import Image

def LandmarkError(gt_lms_list, pred_lms_list, part_inds=None, normalization='centers', showResults=False, verbose=False):
    errors = []
    
    if part_inds is None:
        part_inds = np.arange(60)

    for i in range(len(gt_lms_list)):
        gtLandmarks = gt_lms_list[i]
        predLandmarks = pred_lms_list[i]

        if normalization == 'centers':
            normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        elif normalization == 'corners':
            normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        elif normalization == 'diagonal':
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist = np.sqrt(width ** 2 + height ** 2)
        elif normalization == 'mangaChin':
            normDist = np.linalg.norm(gtLandmarks[0] - gtLandmarks[16])

        #errors = [np.mean(np.sqrt(np.sum((gt_lms_list[i] - pred_lms_list[i])**2,axis=1))) / normDist for i in range(len(gt_lms_list))]
        #mean_errors = np.mean(np.mean(np.sqrt(np.sum((gt_lms_list - pred_lms_list)**2, axis=2)), axis=1) / normDist)
    
        error = np.mean(np.sqrt(np.sum((gtLandmarks[part_inds] - predLandmarks[part_inds])**2,axis=1))) / normDist       
        errors.append(error)
    '''
        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)            
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()
            plt.savefig("../test_results_pdm/{0}".format(path.basename(imageServer.filenames[i])[:-4] + '.png'))
            plt.clf()


        if exportPTS:
            img_menpo = Image(imageServer.img[i])
            img_menpo.landmarks['ecptp_jaw'] = PointCloud(ecptp_jaw)
            img_menpo.landmarks['ecptp_out'] = PointCloud(ecptp_out)
            img_menpo.landmarks['ect'] = PointCloud(ect_lms)
            img_menpo.landmarks['e'] = PointCloud(resLandmarks)
            img_menpo.landmarks['ecp'] = PointCloud(p_pdm_lms)
            img_menpo.landmarks['ecpt'] = PointCloud(pdm_clm_lms)
            mio
    '''
    if verbose:
        #print(errors.shape)
        #print(mean_errors)
        print "Image idxs sorted by error"
        print np.argsort(errors)
    avgError = np.mean(errors)
    print "Average error: {0}".format(avgError)

    return errors


def AUCError(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    print "AUC @ {0}: {1}".format(failureThreshold, AUC)
    print "Failure rate: {0}".format(failureRate)

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()
        plt.savefig("../AUCResult.png")
        plt.clf()

    
# Modified from Face of Art: https://github.com/papulke/face-of-art
import numpy as np
import pickle
import os
from menpo.shape import PointCloud

def pdm_correct(init_shape, pdm_model, part_inds=None):
    """ correct landmarks using pdm (point distribution model)"""
    pdm_model.set_target(PointCloud(init_shape))
    if part_inds is None:
        return pdm_model.target.points
    else:
        return pdm_model.target.points[part_inds]

def feature_based_pdm_corr(lms_init, models_dir, patches=None, pc_opt=[2, 2, 3, 3, 6, 7]):
    
    lms_init_temp = lms_init[:, [1, 0]]

    left_brow_inds = np.arange(47, 52)
    right_brow_inds = np.arange(52, 57)
    left_eye_inds = np.arange(27, 37) # np.append(np.arange(27, 37), 57)
    right_eye_inds = np.arange(37, 47) # np.append(np.arange(37, 47), 58)
    mouth_inds = np.arange(17, 27)
    jaw_line_inds = np.arange(0, 17)
    
    new_lms = np.zeros((60, 2))

    parts = ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'mouth', 'chin']
    part_inds_opt = [left_brow_inds, right_brow_inds, left_eye_inds, right_eye_inds, mouth_inds, jaw_line_inds]
    for i, part in enumerate(parts):
        part_inds = part_inds_opt[i]
        pc = pc_opt[i]
        temp_model = os.path.join(models_dir, part + '_' + str(pc))
        filehandler = open(temp_model, "rb")
        try:
            pdm_temp = pickle.load(filehandler)
        except UnicodeDecodeError:
            pdm_temp = pickle.load(filehandler, fix_imports=True, encoding="latin1")
        filehandler.close()

        #if patches is None:
        part_lms_pdm = pdm_correct(lms_init_temp[part_inds], pdm_temp) 
        #else:
            #part_lms_pdm = w_pdm_correct(
                #init_shape=lms_init_temp[part_inds], patches=patches, pdm_model=pdm_temp, part_inds=part_inds)

        new_lms[part_inds] = part_lms_pdm

    # Nose, Pupils
    new_lms[57:60] = lms_init_temp[57:60]
    
    return new_lms[:, [1, 0]]
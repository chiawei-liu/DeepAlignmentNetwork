# Modified from Face of Art: https://github.com/papulke/face-of-art
import numpy as np
import pickle
import os
from menpo.shape import PointCloud
from menpofit.clm import GradientDescentCLMFitter
from menpo.image import Image
import itertools

def pdm_correct(init_shape, pdm_model, part_inds=None):
    """ correct landmarks using pdm (point distribution model)"""
    pdm_model.set_target(PointCloud(init_shape))
    if part_inds is None:
        return pdm_model.target.points
    else:
        return pdm_model.target.points[part_inds]

def feature_based_pdm_corr(lms_init, models_dir, patches=None, pc_opt=[2, 2, 8, 6, 12, 10]):
    
    lms_init_yx = lms_init[:, [1, 0]]

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
        part_lms_pdm = pdm_correct(lms_init_yx[part_inds], pdm_temp) 
        #else:
            #part_lms_pdm = w_pdm_correct(
                #init_shape=lms_init_yx[part_inds], patches=patches, pdm_model=pdm_temp, part_inds=part_inds)

        new_lms[part_inds] = part_lms_pdm

    # Nose, Pupils
    new_lms[57:60] = lms_init_yx[57:60]
    
    return new_lms[:, [1, 0]]

def clm_correct(clm_model_path, image, lms_init, n_active_components=30, self_targeting=True, opt=None):
    """ tune landmarks using clm (constrained local model)"""
    
    lms_init_yx = lms_init[:, [1, 0]]
    
    image_menpo = Image(image)
    image_menpo.landmarks['PTS'] = PointCloud(lms_init_yx)
    
    filehandler = open(os.path.join(clm_model_path), "rb")
    try:
        part_model = pickle.load(filehandler)
    except UnicodeDecodeError:
        part_model = pickle.load(filehandler, fix_imports=True, encoding="latin1")
    filehandler.close()

    if opt is None:
        # from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment
        part_model.opt = dict()
        part_model.opt['numIter'] = 5
        part_model.opt['kernel_covariance'] = 10
        part_model.opt['sigOffset'] = 25
        part_model.opt['sigRate'] = 0.25
        part_model.opt['pdm_rho'] = 20
        part_model.opt['verbose'] = False
        part_model.opt['rho2'] = 20
        part_model.opt['ablation'] = (True, True)
        part_model.opt['ratio1'] = 0.12
        part_model.opt['ratio2'] = 0.08
        part_model.opt['smooth'] = True
    else:
        part_model.opt = opt

    fitter = GradientDescentCLMFitter(part_model, n_shape=n_active_components)

    image_menpo.rspmap_data = generate_heatmap(image_menpo, lms_init_yx, patch_size=16)[np.newaxis]
    
    if self_targeting:
        lms_tar_yx = lms_init_yx
    else:
        lms_tar_yx = part_model.reference_shape.points

    try:
        fr = fitter.fit_from_shape(image=image_menpo, initial_shape=PointCloud(lms_init_yx), gt_shape=PointCloud(lms_tar_yx))
        w_pdm_clm = fr.final_shape.points
    except:
        w_pdm_clm =  lms_init_yx

    return w_pdm_clm[:, [1, 0]]

def feature_based_clm_corr(clm_models_dir, image, lms_init, n_active_components=30, self_targeting=True, opt=None):
    
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
        part_model_path = os.path.join(clm_models_dir, 'manga_' + part)
    
        #try:
        new_lms[part_inds] = clm_correct(clm_model_path=part_model_path, image=image, lms_init=lms_init[part_inds], n_active_components=n_active_components, self_targeting=self_targeting, opt=opt)
        #except Exception as e:
        #    print('clm failed')
        #    print(e)
        #    new_lms[part_inds] = lms_init[part_inds].copy()

    new_lms[57:60] = lms_init[57:60]
    
    return new_lms

def generate_heatmap(input_img, init_lms, patch_size=16):
    img_shape = input_img.shape
    patch_size = 16
    half_size = int(patch_size / 2)
    offsets = np.array(list(itertools.product(range(-half_size, half_size + 1), range(-half_size, half_size + 1))))
    landmarks = init_lms
    heat_map = np.zeros((landmarks.shape[0], img_shape[0], img_shape[1]), dtype=np.float32)
    
    for i in range(landmarks.shape[0]):
        landmark = landmarks[i]
        intLandmark = landmark.astype('int')
        locations = offsets + intLandmark
        dxdy = landmark - intLandmark
        offsetsSubPix = offsets - dxdy
        vals = 1 / (1 + np.sqrt(np.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6))
        heat_map[i, locations[:, 0], locations[:, 1]] = vals[:]

    return heat_map
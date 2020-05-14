from pdm_clm_functions_manga import feature_based_pdm_corr, clm_correct, feature_based_clm_corr


class MangaPredictor():

    def __init__(self, faceAlignment, pdm_dir='../data/pdm_models/', clm_path='../data/clm_models/'):
        self.faceAlignment = faceAlignment
        self.pdm_dir = pdm_dir
        self.clm_path = clm_path

    def predictFromImageServer(self, imageServer, pdm_dir='../data/pdm_models/', clm_path='../data/clm_models/manga_all'):
        nImgs = len(imageServer.imgs)
        
        jaw_line_inds = np.arange(0, 17)
        left_brow_inds = np.arange(47, 52)
        right_brow_inds = np.arange(52, 57)

        e_list = []
        ect_list = []
        ecp_list = []
        ecpt_list = []
        ecptp_jaw_list = []
        ecptp_out_list = []

        for i in range(nImgs):
            initLandmarks = imageServer.initLandmarks[i]
            gtLandmarks = imageServer.gtLandmarks[i]
            img = imageServer.imgs[i]

            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]

            resLandmarks = initLandmarks
            resLandmarks = self.faceAlignment.processImg(img, resLandmarks)


            # Modified from Face of Art: https://github.com/papulke/face-of-art
            # get landmarks for part-based correction stage
            p_pdm_lms = feature_based_pdm_corr(lms_init=resLandmarks, models_dir='../data/pdm_models/')
            # get landmarks for part-based tuning stage
            try:  # clm may not converge
                pdm_clm_lms = clm_correct(
                    clm_model_path=clm_path, image=img, lms_init=p_pdm_lms, n_active_components=30)
            except:
                pdm_clm_lms = p_pdm_lms.copy()

            # get landmarks ECT
            try:  # clm may not converge
                ect_lms = clm_correct(
                    clm_model_path=clm_model_path, image=img, lms_init=resLandmarks, n_active_components=30)
            except:
                ect_lms = p_pdm_lms.copy()


            # get landmarks for ECpTp_out (tune jaw and eyebrows)
            ecptp_out = p_pdm_lms.copy()
            ecptp_out[left_brow_inds] = pdm_clm_lms[left_brow_inds]
            ecptp_out[right_brow_inds] = pdm_clm_lms[right_brow_inds]
            ecptp_out[jaw_line_inds] = pdm_clm_lms[jaw_line_inds]

            # get landmarks for ECpTp_jaw (tune jaw)
            ecptp_jaw = p_pdm_lms.copy()
            ecptp_jaw[jaw_line_inds] = pdm_clm_lms[jaw_line_inds]

            ecptp_jaw_list.append(ecptp_jaw)  # E + p-correction + p-tuning (ECpTp_jaw)
            ecptp_out_list.append(ecptp_out)  # E + p-correction + p-tuning (ECpTp_out)
            ect_list.append(ect_lms)  # ECT prediction
            e_list.append(resLandmarks)  # init prediction from deep alignent network (E)
            ecp_list.append(p_pdm_lms)  # init prediction + part pdm correction (ECp)
            ecpt_list.append(pdm_clm_lms)  # init prediction + part pdm correction + global tuning (ECpT)

        pred_dict = {
            'E': e_list,
            'ECp': ecp_list,
            'ECpT': ecpt_list,
            'ECT': ect_list,
            'ECpTp_jaw': ecptp_jaw_list,
            'ECpTp_out': ecptp_out_list
        }

        return pred_dict

    def predictWithPartialCLM(self, imageServer, clm_dir='../data/clm_models/', n_active_components=30, self_targeting=True, opt=None):
        nImgs = len(imageServer.imgs)
        ectp_list = []
        for i in range(nImgs):
            initLandmarks = imageServer.initLandmarks[i]
            gtLandmarks = imageServer.gtLandmarks[i]
            img = imageServer.imgs[i]
    
            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]
    
            resLandmarks = initLandmarks
            resLandmarks = self.faceAlignment.processImg(img, resLandmarks)

            ectp_lms = feature_based_clm_corr(
                clm_models_dir=clm_dir, image=img, lms_init=resLandmarks, n_active_components=n_active_components, self_targeting=self_targeting, opt=opt)

            ectp_list.append(ectp_lms)

        return ectp_list

    def predictWithGlobalCLM(self, imageServer, clm_path='../data/clm_models/manga_all', n_active_components=30, self_targeting=True, opt=None):
        nImgs = len(imageServer.imgs)
        ect_list = []
        for i in range(nImgs):
            initLandmarks = imageServer.initLandmarks[i]
            gtLandmarks = imageServer.gtLandmarks[i]
            img = imageServer.imgs[i]

            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]
    
            resLandmarks = initLandmarks
            resLandmarks = self.faceAlignment.processImg(img, resLandmarks)

            ect_lms = clm_correct(
                clm_model_path=clm_path, image=img, lms_init=resLandmarks, n_active_components=n_active_components, self_targeting=self_targeting, opt=opt)

            ect_list.append(ect_lms)

        return ect_list

    def predictPureDAN(self, imageServer):
        nImgs = len(imageServer.imgs)
        e_list = []
        for i in range(nImgs):
            initLandmarks = imageServer.initLandmarks[i]
            gtLandmarks = imageServer.gtLandmarks[i]
            img = imageServer.imgs[i]

            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]
    
            resLandmarks = initLandmarks
            resLandmarks = self.faceAlignment.processImg(img, resLandmarks)

            e_list.append(resLandmarks)

        return e_list

    def predictWithPartialPDM(self, imageServer, pc_opt=None):
        nImgs = len(imageServer.imgs)
        ecp_list = []
        for i in range(nImgs):
            initLandmarks = imageServer.initLandmarks[i]
            gtLandmarks = imageServer.gtLandmarks[i]
            img = imageServer.imgs[i]

            if img.shape[0] > 1:
                img = np.mean(img, axis=0)[np.newaxis]
    
            resLandmarks = initLandmarks
            resLandmarks = self.faceAlignment.processImg(img, resLandmarks)

            p_pdm_lms = feature_based_pdm_corr(lms_init=resLandmarks, models_dir='../data/pdm_models/', pc_opt=pc_opt)
            
            ecp_list.append(p_pdm_lms)

        return ecp_list
    #
    # 
    '''
    def predictFromImage(self, img):
        box = np.array([0, 0, img.shape[2]-1, img.shape[1]-1])
        bestFit = utils.bestFitRect(None, self.faceAlignment..meanShape, box=box)
    '''
from deformation_functions_manga import *
from scipy.interpolate import Rbf
import cv2

def augment_menpo_img_geom(img, landmarks, p_geom=0.):
    """geometric style image augmentation using random face deformations"""
    p_geom = 1. * (np.random.rand() < p_geom) 
    img = img.copy()
    landmarks_temp = landmarks.copy()
    landmarks_temp[:,[0,1]] = landmarks[:,[1,0]]
    if p_geom > 0.5:
        # grp_name = img.landmarks.group_labels[0]
        lms_geom_warp = deform_face_geometric_style(landmarks_temp, p_scale=p_geom, p_shift=p_geom)
        lms_geom_warp[:,[0,1]] = lms_geom_warp[:,[1,0]]
        try:
            img_warp = warpImageTPS(img, landmarks, lms_geom_warp)
            return img_warp, lms_geom_warp
        except Exception as err:
            print ('Error:'+str(err)+'\nUsing original landmarks')
            return img, landmarks
    return img, landmarks

'''
def warp_face_image_tps(img, new_shape):
    """warp image to new landmarks using TPS interpolation"""

    tps = ThinPlateSplines(new_shape, img.landmarks[lms_grp_name])
    try:
        img_warp = img.warp_to_shape(img.shape, tps, mode=warp_mode)
        img_warp.landmarks[lms_grp_name] = new_shape
        return img_warp
    except np.linalg.linalg.LinAlgError as err:
        print ('Error:'+str(err)+'\nUsing original landmarks for:\n'+str(img.path))
        return img
'''
def deform_face_geometric_style(lms, p_scale=0, p_shift=0):
    """ deform facial landmarks - matching ibug annotations of 68 landmarks """

    lms = deform_scale_face(lms.copy(), p_scale=p_scale, pad=0, image_size=112)
    lms = deform_nose(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0)
    lms = deform_mouth(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0)
    lms = deform_eyes(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0)
    return lms
'''
meanShape = np.load("DeepAlignmentNetwork/data/mangaMeanFaceShape.npz")["meanShape"]
trainSet = ImageServer(initialization='rect')
trainSet.PrepareData("DeepAlignmentNetwork", None, meanShape, 0, 100000, False)
trainSet.LoadImages()

# for aumentation
#trainSet.GeneratePerturbations(5, [0.1, 0.1, 20, 0.1])

# for no augmentation
trainSet.CropResizeRotateAll()
trainSet.imgs = trainSet.imgs.astype(np.float32)
'''
#trainSet.NormalizeImages()
#trainSet.Save("./")

'''
class PointsRBF:
    def __init__(self, src, dst):
         xsrc = src[:,0]
         ysrc = src[:,1]
         xdst = dst[:,0]
         ydst = dst[:,1]
         self.rbf_x = Rbf( xsrc, ysrc, xdst, function='thin-plate')
         self.rbf_y = Rbf( xsrc, ysrc, ydst, function='thin-plate')

    def __call__(self, xy):
        x = xy[:,0]
        y = xy[:,1]
        xdst = self.rbf_x(x,y)
        ydst = self.rbf_y(x,y)
        return np.transpose( [xdst,ydst] )

def warpRBF(image, src, dst):
    prbf = PointsRBF( dst, src)
    warped = skimage.transform.warp(image, prbf)
    warped = 255*warped                         # 0..1 => 0..255
    warped = warped.astype(np.uint8)            # convert from float64 to uint8
    return warped
'''
def warpImageTPS(img, src, dst, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT):
	tps = cv2.createThinPlateSplineShapeTransformer()
	src = src.reshape(-1, len(src), 2)
	dst = dst.reshape(-1, len(dst), 2)

	matches=list()
	for i in range(0,len(src[0])):
		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(dst, src, matches)
	new_img = tps.warpImage(img, flags=flags, borderMode=borderMode)
    if np.all(new_img == new_img[0, 0]):
        raise Exception('Warp failed')
	return new_img
import cv2
import numpy as np
from PIL import Image

def calc_score(resfolder, objfolder, index, points):
    score_tp =0
    score_tn =0
    score_fn = 0
    score_fp = 0
    i = 0
    frame = 3 # Size of the surrounding pixels to label corners

    if resfolder == '480p':
        index = '{0:05}'.format(index)
        path = 'DAVIS/Annotations/' + resfolder + '/' + objfolder + '/' + index + '.png'

    else:
        index = '{0:06}'.format(index)
        path = 'KITTI/instances/' + objfolder + '/' + index + '.png'


    im = cv2.imread(path)
    points = points.astype(int)
    classification = np.zeros((points.shape[0]))

    for item in points:
        if np.any(im[points[i, 1]-frame:points[i,1]+frame, points[i, 0]-frame:points[i,0]+frame, 1] > 0) and points[i, 2] == 1:
            score_tp = score_tp+1
            classification[i] = 1
        elif np.any(im[points[i, 1]-frame:points[i,1]+frame, points[i, 0]-frame:points[i,0]+frame, 1] == 0) and points[i, 2] == 1:
            score_fp = score_fp+1
            classification[i] = 2
        elif np.any(im[points[i, 1]-frame:points[i,1]+frame, points[i, 0]-frame:points[i,0]+frame, 1] > 0) and points[i, 2] == 0:
            score_fn = score_fn+1
            classification[i] = 3
        elif np.any (im[points[i, 1]-frame:points[i,1]+frame, points[i, 0]-frame:points[i,0]+frame, 1] == 0) and points[i, 2] == 0:
            score_tn = score_tn+1
            classification[i] = 4
        #else: print(im[points[i, 1], points[i, 0], 1], points[i,2])
        i = i+1
    conf_matrix_temp = [score_tp, score_fp, score_fn, score_tn]

    return conf_matrix_temp, classification
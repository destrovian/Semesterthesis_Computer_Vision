import numpy as np
import cv2
import os

from score import calc_score

from Img_Function_V1 import image_function
from Img_Function_V1 import colour

from skimage.measure import LineModelND, ransac

pathlin = '/home/sem21h12/Downloads/DAVIS/JPEGImages/480p/swan/%05d.jpg'
pathdavis = 'DAVIS/JPEGImages/480p/motorbike/%05d.jpg'
pathkitti = 'KITTI/training/image_02/0003/%06d.png'
path = pathkitti

# save directory strings
parent_directory, directory_name = os.path.split(path)
parent_parent_directory, parent_directory_name = os.path.split(parent_directory)
obj_folder = parent_directory_name
parent_directory, res_folder = os.path.split(parent_parent_directory)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=2000,
                      qualityLevel=0.4,  # used 0.6 for most of the v1-4 ... may use other value later
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

threshold_res = np.zeros((200,14))
for i in range(0,threshold_res.shape[0]):
    temp = np.zeros((1,14))
    conf_matrix, precision, recall, specificity, res_time = image_function(
    path, feature_params, res_folder, obj_folder, lk_params,
    threshold_tandir= 0.001+i*0.001, size_ellipse= 10+i*10,
    algorithm='tandir', ego_motion=False, max_dist=False, noise=False, mult_factor=0.1*i)
    print(conf_matrix)
    print(i)
    temp[0, 0] = 0.001+i*0.001
    temp[0,1:5] = conf_matrix[:]
    temp[0,5] = precision
    temp[0,6] = recall
    temp[0,7] = specificity
    temp[0, 8:14] = res_time[:]
    threshold_res[i,:] = temp[:]
    #n = n+1
    np.savetxt("results/tandir_kitti_0003.csv", threshold_res, delimiter=",")


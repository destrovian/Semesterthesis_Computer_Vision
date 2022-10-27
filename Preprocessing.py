import time

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac
from skimage.transform import FundamentalMatrixTransform

def preprocessing(img):
    temp = cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)
    # temp = cv2.GaussianBlur(temp,(7,7),cv2.BORDER_DEFAULT)
    return temp

"""
# function to find best detected features using brute force
# matcher and match them according to there humming distance
def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    no_of_matches = brute_force.match(des1, des2)

    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
    return no_of_matches

def compute_fundamental_matrix_points(points1, points2, distance1):
    
    Takes in filenames of two input images
    Return Fundamental matrix computes
    using 8 point algorithm
    

    # extract points
    logdirabs = distance1 < 50
    pts1 = points1[logdirabs]
    pts2 = points2[logdirabs]

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    return F, pts1, pts2
"""

def ego_motion_compensation(good_0, good_1, width, height):

    # Estimate the epipolar geometry between the left and right image.
    model, inliers = ransac((good_0, good_1),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=1200)

    # Simple K matrix
    F = model.params
    f = 700
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

    # Essential matrix
    E = np.matmul(np.matmul(np.transpose(K), F), K)

    [points, R, t, mask] = cv2.recoverPose(E, good_0, good_1, K)
    #print("R and t")
    #print(R)
    #print(t)
    return R, t, K, inliers

def point_prediction(points_1, points_2, x_pixels, y_pixels):
    tic = time.perf_counter()
    R, t, K0, inliers = ego_motion_compensation(points_1, points_2, x_pixels, y_pixels)
    toc = time.perf_counter()
    eme = toc-tic
    #points_1 = points_1[np.logical_not(inliers)]

    tic = time.perf_counter()
    K = np.zeros((4, 4))
    K[0:3, 0:3] = K0
    K_inv = np.linalg.pinv(K)
    factor = np.zeros((4, 4))
    factor[3, 3] = 1
    factor[0:3, 0:3] = R
    factor[0, 3] = t[0]
    factor[1, 3] = t[1]
    factor[2, 3] = t[2]

    factor = np.matmul(K, factor)
    factor = np.matmul(factor, K_inv)

    points_pred = np.zeros((points_1.shape[0], 4))
    for i, n in enumerate(points_1):
        temp = np.ones((4, 1))
        temp[0:2, 0] = points_1[i, :]

        x2_pred = np.matmul(factor, temp)
        x2_pred = x2_pred / x2_pred[2, 0]
        points_pred[i,:] = x2_pred.transpose()

    toc = time.perf_counter()
    emc = toc-tic

    return points_pred[:,0:2], inliers, eme, emc
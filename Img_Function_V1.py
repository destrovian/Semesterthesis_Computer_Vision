import numpy as np
import cv2
import os
from Preprocessing import preprocessing
from Preprocessing import compute_fundamental_matrix_points
from score import calc_score
from Intersect import intersect_labels
from Intersect import findIntersection
from Intersect import logical_label
from Intersect import ellipsoid_distance
from Intersect import RayTracing_point_cloud
from Intersect import is_escape_point
import time

from Preprocessing import preprocessing
from Preprocessing import ego_motion_compensation
from Preprocessing import point_prediction

def colour(counter): #colour is in BGR
    if counter == 1:
        return (255,255,255)
    elif counter ==2:
        return (255,0,0)
    elif counter ==3:
        return (0,255,0)
    elif counter ==4:
        return (0,0,255)


def image_function(path, feature_params, res_folder, obj_folder,
                   lk_params, threshold_tandir, size_ellipse,
                   algorithm, ego_motion, max_dist, noise, mult_factor):

    tot_time = np.zeros((8))
    res_time = np.zeros((8))

    # Initialize VideoCapture
    cap = cv2.VideoCapture(path)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.cvtColor(old_frame + 1, cv2.COLOR_BGR2GRAY)

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

    # create confusion matrix
    conf_matrix = np.zeros((1, 4))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    iter = 0

    while (1):
        ret, frame = cap.read()

        tictot = time.perf_counter()  # logging of entire loop
        tic = time.perf_counter()

        frame2_gray = cv2.cvtColor(frame + 2, cv2.COLOR_BGR2GRAY)

        # lets do some Gaussian Blurring
        if iter == 0:
            old_gray = preprocessing(old_gray)
            frame1_gray = preprocessing(frame1_gray)

        frame2_gray = preprocessing(frame2_gray)

        toc = time.perf_counter()

        res_time[0] =  toc-tic #logging of pre-processing step

        # Recalculate corners cause they may disappear due to camera movement
        # maybe dont do this every image but every few to better extrapolate direction + outliers
        # if iter%3==0: # every iter frame the initial points are recalculated

        tic = time.perf_counter()

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params, useHarrisDetector = False)


        #lets adapt the search area already so no unneccessary comp.power is used
        if res_folder == "image_02":
            logical = np.zeros(p0.shape[0])
            for i in range(0,p0.shape[0]):
                if p0[i,0,1] > 150: logical[i]=1
                elif p0[i,0,1] > 275 and p0[i,:,0]>480 and p0[i,0,0]< 750: logical[i] =1
            p0 = p0[logical.astype(bool),:,:]


        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame1_gray, p0, None, **lk_params)
        p2, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame2_gray, p1, None, **lk_params)

        # Select good points
        good_0 = p0[st == 1]
        good_1 = p1[st == 1]
        good_2 = p2[st == 1]

        toc = time.perf_counter()
        res_time[1] = toc-tic #logging of feature extraction


        if ego_motion == False:
            direction1 = good_1 - good_0
            direction2 = good_2 - good_1

        # egomotion compensation -> point prediction
        if ego_motion == True:
            #tic = time.perf_counter()
            good_1_pred, inliers1, eme, emc = point_prediction(good_0, good_1, frame1_gray.shape[0],
                                                               frame1_gray.shape[1])

            res_time[6] = res_time[6]+eme
            res_time[7] = res_time[7]+emc

            inliers = np.logical_not(inliers1)

            good_2_pred, inliers2, eme, emc = point_prediction(good_1, good_2, frame1_gray.shape[0],
                                                               frame1_gray.shape[1])

            res_time[6] = res_time[6] + eme
            res_time[7] = res_time[7] + emc

            # now lets caclculate the actual movement of the features:
            direction1 = good_1 - good_1_pred
            direction2 = good_2 - good_2_pred
            #toc = time.perf_counter()
            #res_time[2] = toc-tic

            #correct all directions and features:
            good_0 = good_0[inliers]
            good_1 = good_1[inliers]
            direction1 = direction1[inliers]
            direction2 = direction2[inliers]


        if noise == True:
            direction1 = direction1 + (2*np.random.rand(direction1.shape[0], direction1.shape[1]-1)*mult_factor)
            direction2 = direction2 + (2*np.random.rand(direction1.shape[0], direction1.shape[1]-1)*mult_factor)

        if max_dist == True: #TODO rewrite this to area of interest function for kitti
            # lets remove the outliers and missed measurements
            distance = np.linalg.norm(direction1, axis=1)
            logical_abs_upper = distance < 50
            logical_abs_lower = distance > 0
            logical_abs = logical_abs_lower * logical_abs_upper

            direction1 = direction1[logical_abs, :]
            direction2 = direction2[logical_abs, :]
            good_0 = good_0[logical_abs]
            good_1 = good_1[logical_abs]

        if algorithm == 'tandir': # TANDIR of directional change
            tic = time.perf_counter()
            tandir = direction2[:, 1] / direction2[:, 0] - \
                     direction1[:, 1] / direction1[:, 0]  # calculate angle difference of 2 directions

            """
            # TODO compensate for directional changes according to optical flow length
            for i in range(0, direction2.shape[0]):
                lin_factor = np.linalg.norm(direction2)/np.linalg.norm(direction3)
                if np.linalg.norm(direction2)/np.linalg.norm(direction3) < 1:
                    tandir[i] = tandir[i] * 1 / (lin_factor * lin_factor)
                else: tandir[i] = tandir[i] * lin_factor * lin_factor
            """

            tandir = tandir / np.linalg.norm(tandir)  # doesnt work without - const value fails
            threshold = threshold_tandir
            logical_tandir = np.abs(tandir) > threshold
            # tandir = tandir[logical]
            # print("Sum of detected points:", np.sum(logical))
            logical = logical_tandir

            # Deleta all unwanted points
            good_new = good_1[logical]
            good_old = good_0[logical]
            toc = time.perf_counter()
            res_time[3] = toc-tic

        if algorithm == 'raytrace': # RAYTRACING of optical flow into vanishing point
            tic = time.perf_counter()
            #print(good_0.shape)
            esc_point_cloud = RayTracing_point_cloud(good_0, direction1, ellipsoid=size_ellipse)
            toc = time.perf_counter()
            res_time[3] = toc - tic

            good_new = esc_point_cloud
            good_old = esc_point_cloud

            logical = np.zeros((good_0.shape[0]))

            for i in range(good_0.shape[0]):
                for n in range(esc_point_cloud.shape[0]):
                    if good_0[i,0] == esc_point_cloud[n,0] and good_0[i,1] == esc_point_cloud[n,1]:
                        logical[i] = 1
            logical = logical.astype(bool)

            # create a label array for classification results


        # Prepare points plus logical for confusion matrix:
        tic = time.perf_counter()
        corners = np.zeros((logical.shape[0], 4))
        corners[:, 0] = good_0[:, 0]
        corners[:, 1] = good_0[:, 1]
        corners[:, 2] = logical

        # Scoring system based on confusion matrix
        conf_matrix_temp, classification = calc_score(res_folder, obj_folder, iter, corners)
        conf_matrix = conf_matrix + conf_matrix_temp
        corners[:,3] = classification
        classification_mask = classification[logical]
        #print(conf_matrix)
        toc = time.perf_counter()
        res_time[4] = toc-tic


        # draw the tracks

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            #frame = cv2.circle(frame, (int(a), int(b)), 8, (255, 0, 0), -1)
            #frame = cv2.circle(frame, (int(a), int(b)), 8, colour(classification_mask[i]), -1)

        img = cv2.add(frame, mask)
        toctot = time.perf_counter()
        res_time[5] = toctot-tictot


        #cv2.imwrite('tandir_kitti_' +str(iter) + '.jpg', img)
        #cv2.imshow('frame', img)
        #k = cv2.waitKey(0) & 0xff
        #if k == 27:
        #   break

        #print([good_new.shape, good_old.shape])


        # Now update the previous frame and previous points
        old_gray = frame1_gray
        frame1_gray = frame2_gray

        iter = iter + 1
        #print("Iteration", iter)
        # break condition for loops:
        if iter == 40: break
        #print("Time spent: " , res_time)
        tot_time = tot_time + res_time

    #print(tot_time)
    cv2.destroyAllWindows()
    cap.release()

    precision = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[0,2])
    specificity = conf_matrix[0,3]/(conf_matrix[0,3] + conf_matrix[0,1])

    return conf_matrix, precision, recall, specificity, tot_time

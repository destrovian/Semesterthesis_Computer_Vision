import numpy as np


def logical_label(labels):
    labels = labels +1
    unique, counts = np.unique(labels, return_counts=True)
    y = unique[counts > 0.8*counts.max()].astype(int)
    result = labels == y.any()
    return result


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def ellipsoid_distance(ellipse_center, flowdir, point_intersect, ellipsoid_size, bbox_height):
    # TODO have no idea how big this should be so probably run with a factor linspace
    flowdir = [flowdir[0], flowdir[1]]
    flowdir = flowdir / np.linalg.norm(flowdir)
    point_intersect = np.asarray(point_intersect)
    point_array = [point_intersect[0,0],point_intersect[1,0]]
    pole_distance = [element * bbox_height for element in flowdir]
    pole_1 = ellipse_center + pole_distance
    pole_2 = ellipse_center - pole_distance

    #print(point_array)
    #print(np.linalg.norm(point_array - pole_2) + np.linalg.norm(point_array - pole_1))
    if np.linalg.norm(point_array - pole_2) + np.linalg.norm(point_array - pole_1) < ellipsoid_size:
        #print('True')
        return True
    else:
        #print('False')
        return False


def is_escape_point(point_intersect, point1, point2, opticalflow1, opticalflow2):
    if ((point_intersect - point1)/opticalflow1 < 0).all() and ((point_intersect - point2)/opticalflow2 < 0).all():
        return True
    else:
        return False


def RayTracing_point_cloud(points, opticalflow, ellipsoid):
    cloud = np.empty((1,8)) # esc_point_x, esc_point_y, dir_x, dir_y, x1, y1, x2, y2
    ref_ellipse = np.empty((1,2)) # the reference matrix that new entries are compared to
    factor = ellipsoid/4
    ellipsoid_size = ellipsoid

    for n in range(0,points.shape[0]):
        for m in range(1+n,points.shape[0]-1):
            p_intersect = findIntersection(points[n,0],points[n,1],points[n,0]+opticalflow[n,0],points[n,1]+opticalflow[n,1],
                             points[m, 0], points[m, 1], points[m, 0] + opticalflow[m, 0], points[m, 1] + opticalflow[m, 1])
            escape_point = is_escape_point(p_intersect, points[n,:], points[m,:], opticalflow[n,:], opticalflow[m,:])

            if escape_point == True: # check if raytracing actually ends in escape point and not somewhere else
                temp = np.zeros((1,8))
                temp[0,0] = p_intersect[0]
                temp[0,1] = p_intersect[1]
                temp[0,2:4] = (opticalflow[n,:]+opticalflow[m,:])/np.linalg.norm((opticalflow[n,:]+opticalflow[m,:]))
                temp[0,4:6] = points[n,:]
                temp[0,6:8] = points[m,:]
                for i in range(0,ref_ellipse.shape[0]):
                    pole1 = ref_ellipse[i] + factor * temp[0,2:4]
                    pole2 = ref_ellipse[i] + factor * temp[0,2:4]
                    if np.linalg.norm(p_intersect - pole2) + np.linalg.norm(p_intersect - pole1) < ellipsoid_size:
                        temp[0,0] = ref_ellipse[i,0]
                        temp[0,1] = ref_ellipse[i,1]
                    elif i == ref_ellipse.shape[0]-1:
                        ref_ellipse = np.append(ref_ellipse,[temp[0,0:2]],axis=0)

                cloud = np.append(cloud, temp, axis=0)

    # make all doubles go away:
    if cloud.shape[0] > 1:
        cloud_unique, count = np.unique(cloud[1:,0:2], axis=0, return_counts=True)
    else: cloud_unique, count = np.unique(cloud[0:,0:2], axis=0, return_counts=True)
    #starting at 1: because for some gawd daym reason np.empty is NOT empty when iterated REEEEEEEEEEEEE

    cloud_unique = cloud_unique[count >= 0.5 * np.max(count)] # TODO find threshold

    logical = np.zeros((cloud.shape[0]))

    # there must be a better way what the heck...
    for i in range(0,cloud.shape[0]):
        for n in range(0,cloud_unique.shape[0]):
            if np.all(cloud[i,0:2] == cloud_unique[n,:]):
                logical[i] = 1

    cloud = cloud[logical.astype(bool)]

    final_points = cloud[:,6:8]
    final_points = np.append(final_points, cloud[:,4:6], axis=0)
    final_points = np.unique(final_points, axis=0)

    return final_points




def intersect_labels(points, opticalflow, threshold, ellipse): # TODO fix this... cause i think is broken
    factor = 1000000
    counter = 0
    bboxes = points.shape[0]
    refindex = np.zeros((bboxes, 1)).astype(int)
    bbox = np.zeros((bboxes,4))
    print(points.shape)

    # Use 0,1 index points to create first BBox center
    i = 1
    n = 0
    npoints = points.shape[0]
    first_bbox = findIntersection(points[i, 0] - factor * opticalflow[i, 0],
                                   points[i, 1] - factor * opticalflow[i, 1],
                                   points[i, 0] + factor * opticalflow[i, 0],
                                   points[i, 1] + factor * opticalflow[i, 1],
                                   points[refindex[n], 0] - factor * opticalflow[refindex[n], 0],
                                   points[refindex[n], 1] - factor * opticalflow[refindex[n], 1],
                                   points[refindex[n], 0] + factor * opticalflow[refindex[n], 0],
                                   points[refindex[n], 1] + factor * opticalflow[refindex[n], 1])
    bbox[n,0:2] = first_bbox
    bbox[n,2:] = opticalflow[i, :] / np.linalg.norm(opticalflow[i, :])

    # Loop over 1 (using 0 as reference) until all points are labeled
    labels = np.zeros(points.shape[0])
    for i in range(1,points.shape[0]):
        for n in range(0,np.count_nonzero(refindex[:,0])+1):
            p_intersect = findIntersection(points[i, 0] - factor * opticalflow[i, 0],
                                           points[i, 1] - factor * opticalflow[i, 1],
                                           points[i, 0] + factor * opticalflow[i, 0],
                                           points[i, 1] + factor * opticalflow[i, 1],
                                           points[refindex[n], 0] - factor * opticalflow[refindex[n], 0],
                                           points[refindex[n], 1] - factor * opticalflow[refindex[n], 1],
                                           points[refindex[n], 0] + factor * opticalflow[refindex[n], 0],
                                           points[refindex[n], 1] + factor * opticalflow[refindex[n], 1])

            #if np.abs(bbox[n,1]-p_intersect[1]) < bbox_threshold:
            if ellipsoid_distance(bbox[n,0:2], opticalflow[i,:],p_intersect, ellipse, threshold):
                # and np.abs(bbox[n,0]-p_intersect[0]) < bbox_threshold: # currently only 1D to see if it works better
                bbox[n,1]= bbox[n,1]+(p_intersect[1]-bbox[n,1])/npoints # update the bbox center for each point labelled to it
                bbox[n,0]= bbox[n,0]+(p_intersect[0]-bbox[n,0])/npoints
                bbox[n, 2:] = bbox[n, 2:] + opticalflow[i, :] / np.linalg.norm(opticalflow[i, :])
                labels[i] = n
                refindex[n] = i
                break

            elif n == np.count_nonzero(refindex[:,0]):
                bbox[counter,0] = p_intersect[0]
                bbox[counter,1] = p_intersect[1]
                bbox[counter, 2:] = opticalflow[i,:]/np.linalg.norm(opticalflow[i,:])
                refindex[counter] = i
                counter = counter + 1
                labels[i] = counter
                break

    return labels.astype(int)

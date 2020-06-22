import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import cv2
from albumentations import Rotate
from scipy import ndimage
import imutils
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import medial_axis, skeletonize, thin, remove_small_objects
from rdp import rdp
import rdp
from csaps import csaps

def get_points(skeleton_image):
    cnts = cv2.findContours(skeleton_image.copy().astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def num_to_str(contour):
    str_list = list()
    for c in contour:
        str_list.append(str(c[0][0])+','+str(c[0][1]))
    return str_list



def get_unique(contour):
    cont = contour.copy()
    str_list = num_to_str(cont)
    index_dict = dict()
    for i in range(len(str_list)):
        if str_list[i] not in index_dict.keys():
            index_dict[str_list[i]] = i
    new_dict = dict((val,key) for key,val in index_dict.items())
    sub_cont = list()
    keys = list(new_dict.keys())
    keys.sort()
    sub_cont.append(list())
    last_ind = 0
    sub_cont[last_ind].append(new_dict[keys[0]])
    for i in range(len(keys)-1):
        if keys[i+1]-keys[i]!=1:
            last_ind+=1
            sub_cont.append(list())
            sub_cont[last_ind].append(new_dict[keys[i+1]])
        else:
            sub_cont[last_ind].append(new_dict[keys[i+1]])
    num_cont = sub_cont.copy()
    for l in range(len(num_cont)):
        for s in range(len(num_cont[l])):
            nums = num_cont[l][s].split(',')
            x = int(nums[0])
            y = int(nums[1])
            num_cont[l][s] = [x,y]
    return num_cont

def get_all_unique(all_contours):
    all_unique = list()
    for cont in all_contours:
        u = get_unique(cont)
        for s in u:
            if len(s)>3:
                all_unique.append(s)
    return all_unique


def makeImage(uniqueCont,shape):
    zeros = np.zeros(shape,dtype=bool)
#     print(len(uniqueCont))
    for pixel in uniqueCont:
#         print(pixel[1],pixel[0])
#         print(pixel)
        x = pixel[1]
        y = pixel[0]
        zeros[x,y] = True
#     plt.imshow(zeros,cmap='gray')
    return zeros

def getXY(segment):
    x = list()
    y = list()
    for p in segment:
        x.append(p[0])
        y.append(p[1])
    return x,y


def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

def get_turning_points_contour(contour, tolerance, min_angle):
    points = np.array(contour)
    simplified = np.array(rdp.rdp(points.tolist(), tolerance))
    sx, sy = simplified.T
    # compute the direction vectors on the simplified curve
    directions = np.diff(simplified, axis=0)
    theta = angle(directions)
    # Select the index of the points with the greatest theta
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta>min_angle)[0]+1
    return np.array([sx[idx], sy[idx]])

def get_turning_points(image, tolerance, min_angle):
    contours = get_points(image)
    corner_points = []
    for contour in contours:
        uniques = get_unique(contour)
        for unique in uniques:
            corner_point = get_turning_points_contour(unique, tolerance, min_angle)
            if corner_point.size != 0:
                corner_points.append(corner_point.T)
                
    points = []
    for pt in corner_points:
        for p in pt:
            points.append(p)
    points = np.array(points)
    return points




def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5



def _curve_length(x, y):
    """
    calculates the length(distance) of the given curve, iterating from point to point.
    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the curve length
    """
    distance = 0
    for i in range(0, len(x) - 1):
        distance += _distance_2p(x[i], y[i], x[i + 1], y[i + 1])
    return distance



def _chord_length(x, y):
    """
    distance between starting and end point of the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the chord length of the given curve
    """
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])



def get_smooth_coor(cont1):
    x,y = getXY(cont1)
    index = list(range(len(x)))
    index2 = np.linspace(index[0],index[-1],5000)
    xs = csaps(index,x, index2, smooth=0.01)
    ys = csaps(index,y, index2, smooth=0.01)
    return xs,ys

def sd_theta(x,y):
    m1 = 0
    diffx = np.diff(x).astype(np.float64)
    diffy = np.diff(y).astype(np.float64)
    diffx+=1e-10
    dy = np.divide(diffy,diffx)
    slopes = list()
    theta = list()
    alpha_x = x
    g = alpha_x[0]
    is_valid = False
    for k in alpha_x:
        if k!=g:
            is_valid = True
            break
    if is_valid:
        for i in range(len(x)-1):
            tan_line = (x-x[i])*dy[i] + y[i]
            coeffs = np.polyfit(x,tan_line,1)
            slopes.append(coeffs[0])
            m2 = coeffs[0]
            t = np.arctan((m2-m1)/(1+m1*m2)) * (180/np.pi)
            theta.append(t)
        return np.std(theta) , slopes ,theta
    else:
        for i in range(len(x)-1):
            slopes.append(1e10)
            m2 = 1e10
            t = np.arctan((m2-m1)/(1+m1*m2)) * (180/np.pi)
            theta.append(t)

        return 0,slopes,theta


def tortousity(cont1):
    smoothX,smoothY = get_smooth_coor(cont1)
    
    dx = np.diff(smoothX)
    dy = np.diff(smoothY)

    dx2 = (dx[0:-1] + dx[1:])/2
    dy2 = (dy[0:-1] + dy[1:])/2

    dx = dx[:-1]
    dy = dy[:-1]

    k = np.divide((np.multiply(dx ,dy2) - np.multiply(dx2 ,dy)) ,  np.power((np.square(dx) + np.square(dy)),3/2))
    
    float_ip = list()
    
    inflection_index = list()
    
    for i in range(len(k)-1):
        if k[i]*k[i+1]<0:
           float_ip.append([smoothX[i+1],smoothY[i+1]])
           inflection_index.append(i+1)
    # used for plotting:
	# print(float_ip)
    actual_ip = list()
    for fip in float_ip:
        min_dist = _distance_2p(fip[0],fip[1],cont1[0][0],cont1[0][1])
        minx = cont1[0][0]
        miny = cont1[0][1]
        for i in range(len(cont1)):
            d = _distance_2p(fip[0],fip[1],cont1[i][0],cont1[i][1])
	#   print(cont1[i][0],cont1[i][1])
	#   print(d)
            if d<=min_dist:
                minx = cont1[i][0]
                miny = cont1[i][1]
                min_dist = d
        actual_ip.append([minx,miny])
#     print(actual_ip)
    n = len(inflection_index) +1
#     print(n)
    total_cl = _curve_length(smoothX,smoothY)
#     print(total_cl)
#     print(inflection_index)
    tort = (n-1)/(n*total_cl)
    dm = 0
    if n==1:
        al = _curve_length(smoothX,smoothY)
        cl = _chord_length(smoothX,smoothY)
        dm+=al/cl-1
    #     print(dm)
    elif n==2:
        ind = inflection_index[0]
        al = _curve_length(smoothX[:ind],smoothY[:ind])
        cl = _chord_length(smoothX[:ind],smoothY[:ind])
        dm+=al/cl
        al = _curve_length(smoothX[ind:],smoothY[ind:])
        cl = _chord_length(smoothX[ind:],smoothY[ind:])
        dm+=al/cl-1
    #     print(dm)
    else:
        ind = inflection_index[0]
        al = _curve_length(smoothX[:ind],smoothY[:ind])
        cl = _chord_length(smoothX[:ind],smoothY[:ind])
        dm+=al/cl-1
#         print(smoothX[0],smoothX[ind],al,cl,dm)
        for ind1 , ind2 in zip(inflection_index[:-1],inflection_index[1:]):
            al = _curve_length(smoothX[ind1:ind2],smoothY[ind1:ind2])
            cl = _chord_length(smoothX[ind1:ind2],smoothY[ind1:ind2])
            dm+=al/cl-1
#             print(smoothX[ind1],smoothX[ind2],al,cl,dm)
        last = inflection_index[-1]
        al = _curve_length(smoothX[last:],smoothY[last:])
        cl = _chord_length(smoothX[last:],smoothY[last:])
        dm+=al/cl-1
#         print(smoothX[last],smoothX[last],al,cl,dm)
    tort*=dm
    
    
    turning_points_num = 0
    turning_points = list()
    
    segment = makeImage(cont1,(210,210))
    
    zeros = np.zeros((210,210))
    
    zeros[segment] = 1
    
    turning_points = get_turning_points(zeros, tolerance = 2, min_angle = np.pi*0.15)
    turning_points_num = len(turning_points)
    
    sd_angle,_,_ = sd_theta(smoothX,smoothY)
    
    return tort, dm , actual_ip , turning_points_num, sd_angle
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

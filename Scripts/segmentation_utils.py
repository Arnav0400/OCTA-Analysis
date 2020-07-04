import numpy as np
import pandas as pd
import math
import cv2
from skimage.morphology import medial_axis, skeletonize, thin, remove_small_objects

def clean_image(img):
    """
    To delete name and centercrop
    """
    H,W = img.shape
    img = img[:W,...]
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def crop_image(img):
    H,W = img.shape
    img = img[:W,...]
    img = img[W//4:W//4+W//2,W//4:W//4+W//2]
    return img

def crop_image_with_padding(img, w):
    H,W = img.shape
    img = img[:W,...]
    img = img[W//4-w//2:W//4+W//2+w//2,W//4-w//2:W//4+W//2+w//2]
    return img

def remove_boundary(img, w):
    H, W = img.shape
    img = img[w//2:-w//2+1,w//2:-w//2+1]
    return img

def circular_crop(img):
    H, W = img.shape
    mask = np.zeros((H,W), np.uint8)
    mask = cv2.circle(mask,(H//2, W//2),H//2,(255,255,255),thickness=-1)
    img = img*mask
    return img

def get_line_arrays(w, l, angle):
    window = (np.ones((w,w))*0).astype('uint8')
    mid_point = w//2
    l = l//2
    if((angle<math.pi/4) or (angle>math.pi/2+math.pi/4)):
        start_point = (int(mid_point+l), int(mid_point-l*math.tan(angle)))
        end_point = (int(mid_point-l), int(mid_point+l*math.tan(angle)))
    elif(angle==math.pi/2):
        start_point = (mid_point, int(mid_point+l))
        end_point = (mid_point, int(mid_point-l))
    elif(angle==math.pi/4):
        start_point = (int(l+mid_point), int(mid_point+l))
        end_point = (int(-l+mid_point), int(mid_point-l))
    elif(angle==math.pi/2+math.pi/4):
        start_point = (int(-l+mid_point), int(mid_point+l))
        end_point = (int(l+mid_point), int(mid_point-l))
    else:
        start_point = (int(l/math.tan(angle)+mid_point), int(mid_point+l))
        end_point = (int(mid_point-l/math.tan(angle)), int(mid_point-l))
    color = 1
    thickness = 1
    window = cv2.line(window, start_point, end_point, color, thickness)
    return window
    
def lineseg(image, w, l):
    img = image.copy()
    H, W = img.shape
    
    for i in range(w//2, W-w//2):
        for j in range(w//2, H-w//2):
            window = img[i-w//2:i+w//2+1, j-w//2:j+w//2+1]
            line_responses = []
            for k in range(12):
                window_k = get_line_arrays(w, l, math.pi/12*k)
                line_response = np.sum(window_k*window)/np.sum(window_k)
                line_responses.append(line_response)
            img[i, j] = max(line_responses) - np.mean(window)
    return img

def seg_img_multi(img, w_l = [(15, 3), (15, 7), (15, 11), (15, 15), (11, 3), (11, 7), (11, 11), (7, 3), (7,7), (3,3)]):
    temp = clean_image(img)
    img = np.zeros_like(crop_image(temp), dtype = float)
    for w_l_i in w_l:
        img += remove_boundary(lineseg(crop_image_with_padding(temp, w_l_i[0])/255, w_l_i[0], w_l_i[1]), w_l_i[0])
    return img/(len(w_l)+1)

def knn(img_seg):
    img = img_seg*255
    # img = cv2.fastNlMeansDenoising(img,None,15,7,11) #mean denoising
    vectorized = img.reshape((-1,1))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER,20000, 0.0001)
    K = 2
    attempts=100
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

def standardize(img):
    return (img-img.min()/img.max()-img.min())

def segment(gray_image):
    img_seg = seg_img_multi(gray_image)
    img_segn = standardize(img_seg)
    img_knn = knn(img_segn)
    image = remove_small_objects((img_knn>=np.unique(img_knn)[1]).astype(bool), min_size=100, connectivity=0).astype(float)
    skel_img = skeletonize(image).astype(int)
    final_img = circular_crop(skel_img)
    return img_segn, img_knn, image, skel_img, final_img 
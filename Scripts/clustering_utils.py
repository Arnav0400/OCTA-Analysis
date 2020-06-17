import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
import random
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
from torchvision import datasets, models, transforms, utils
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm_notebook as tqdm
import albumentations as aug
from albumentations import (HorizontalFlip, VerticalFlip, Normalize, Resize, Rotate, Compose)
from albumentations.pytorch import ToTensor
from ranger import Ranger
import sys
import glob
from scipy import ndimage
from skimage.filters import hessian
from skimage.morphology import medial_axis, skeletonize, thin, remove_small_objects

seed = 23
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNetTrunk(nn.Module):
  def __init__(self):
    super(ResNetTrunk, self).__init__()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion,
                       track_running_stats=self.batchnorm_track),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample,
                        track_running_stats=self.batchnorm_track))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
        block(self.inplanes, planes, track_running_stats=self.batchnorm_track))

    return nn.Sequential(*layers)

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
        
__all__ = ["ClusterNet5g"]


class ClusterNet5gTrunk(ResNetTrunk):
  def __init__(self):
    super(ClusterNet5gTrunk, self).__init__()

    self.batchnorm_track = True

    block = BasicBlock
    layers = [3, 4, 6, 3]

    in_channels = 1
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x, penultimate_features=False):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    if not penultimate_features:
      # default
      x = self.layer4(x)
      x = self.avgpool(x)

    x = x.view(x.size(0), -1)

    return x


class ClusterNet5gHead(nn.Module):
  def __init__(self, num_sub_heads, num_classes):
    super(ClusterNet5gHead, self).__init__()

    self.batchnorm_track = True

    self.num_sub_heads = num_sub_heads
    self.num_classes = num_classes
    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(512 * BasicBlock.expansion, self.num_classes),
      nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in range(self.num_sub_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results


class ClusterNet5g(ResNet):
  def __init__(self, num_sub_heads = 5, num_classes = 5):
    # no saving of configs
    super(ClusterNet5g, self).__init__()

    self.batchnorm_track = True

    self.trunk = ClusterNet5gTrunk()
    self.head = ClusterNet5gHead(num_sub_heads, num_classes)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False, trunk_features=False,
              penultimate_features=False):
    x = self.trunk(x, penultimate_features=penultimate_features)

    if trunk_features:  # for semisup
      return x

    x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
    return x

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  p_i_j = compute_joint(x_out, x_tf_out)
  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_j[(p_j < EPS).data] = EPS
  p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

  loss_no_lamb = loss_no_lamb.sum()

  return loss, loss_no_lamb


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j

def clean_image(img):
    """
    To delete name and centercrop
    """
    H,W,_ = img.shape
    img = img[:W,...]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[W//4:W//4+W//2,W//4:W//4+W//2]
    return img

def extract_vessels(img_orig):
    kernel = np.ones((2,2),np.uint8) #kernel for erosion and dilation

    image = (hessian(img_orig)*255).astype('uint8') #ridge detection

    image = remove_small_objects(image.astype(bool), min_size=64, connectivity=0).astype(float)

    image = cv2.dilate(image,kernel,iterations = 3) #to join disconnected 
    image = cv2.erode(image,kernel,iterations = 3)  #to join disconnected
    image = cv2.dilate(image,kernel,iterations = 1) #to join disconnected
    image = cv2.erode(image,kernel,iterations = 1)  #to join disconnected
    image = cv2.dilate(image,kernel,iterations = 1) #to join disconnected

    #Use any one below
    skel = skeletonize(image)
    # skel = thin(image)
    # med, distance = medial_axis(image, return_distance=True)
    # # skel = med*distance

    #To get the circular part
    mask = np.zeros_like(image, np.uint8)
    H,W = image.shape
    mask = cv2.circle(mask, (H//2,W//2), H//2, (255,255,255),thickness=-1)
    masked_data = cv2.bitwise_and(skel*255, skel*255, mask=mask)
    return masked_data

class Turosity_Single_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, shape = 420, train = True):
        self.transforms = get_transforms(train)
        self.shape = shape
        self.path = path
        
    def __getitem__(self, idx):
        if idx<len(self.path):
#             print(idx)
            img = cv2.imread(self.path[idx])
            if self.shape!=None:
                img = cv2.resize(img,(self.shape,self.shape))
            img = clean_image(img)
            img = extract_vessels(img)
            img = self.transforms(img)
            return img[np.newaxis,...]
        

    def __len__(self):
        return len(self.path)

def get_transforms(phase):
    if phase:
        list_transforms = Compose(
            [
             VerticalFlip(prob=0.5),
             HorizontalFlip(prob=0.5),
             Rotate(prob=1.0, limit=360),
             ToTensor()
            ]
        )
    else:
        list_transforms = ToTensor()
    return list_transforms

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class Rotate:
    def __init__(self, limit=360, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
          
            height, width = img.shape[:2]
            image_center = (width/2, height/2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

            abs_cos = abs(rotation_mat[0,0])
            abs_sin = abs(rotation_mat[0,1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]
            img = img.astype('float32')
            img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
            h, w = img.shape[:2]
            dy = (h - height) // 2
            dx = (w - width) // 2

            y1 = dy
            y2 = y1 + height
            x1 = dx
            x2 = x1 + width
            img = img[y1:y2, x1:x2]
        return img

class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img =cv2.flip(img, 1)
        return img
    
class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
        return img

class ToTensor:
    def __init__(self):
        self.prob=1

    def __call__(self, image):
        img = image
        img = torch.from_numpy(np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        return img
    
def provider(path = glob.glob('FAZ_Tortuosity/*/*'), shape = 420, batch_size=32, num_workers=4, train = True):
    dataset = Turosity_Single_Image_Dataset(path,shape,train)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False #train,
    )

    return dataloader
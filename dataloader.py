import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
import pandas as pd
from skimage import transform, io, img_as_float, exposure
import matplotlib.pyplot as plt
from torchvision.io import read_image
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,Resize,
    RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform,  HueSaturationValue,
    RGBShift,  Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    GaussNoise,OpticalDistortion,RandomSizedCrop, RandomResizedCrop, Normalize
)

def build_ts_transformations():
    AUGMENTATIONS = Compose([
      Resize(height=224, width=224),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
      Normalize()
    ])
    return AUGMENTATIONS


class Eye_Dataset(Dataset):
  def __init__(self, images_path, file_path, imagetype, train):
    self.imagetype = imagetype
    self.images_path = images_path
    self.csv = pd.read_csv(file_path)
    self.image_list = self.csv[:]['ImageName']
    self.label_list = np.array(self.csv.drop(['ImageName'], axis=1))
    self.train = train
    if train == True:
      self.augment = build_ts_transformations()
    else :
      self.augment = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    """
    print(self.image_list)
    print(self.label_list)
    print("images path :",images_path) 
    print("imagetype :",imagetype) 
    ("csv :",file_path) 
    print("train :",train)
    """

  def __getitem__(self,index):
    imagename = self.image_list[index]
    img_path = self.images_path + imagename + "."+self.imagetype
    imageData = cv2.imread( self.images_path + imagename + "."+self.imagetype)
    if imageData is None:
      imageData = cv2.imread( self.images_path + imagename + ".tif")
    imageLabel = torch.FloatTensor(self.label_list[index])
    if self.train == True :
      
      augmented_img = self.augment(image = imageData, mask = imageData)
      student_img = augmented_img['image']
      teacher_img = augmented_img['mask']
      """
      fig, axes = plt.subplots(1, 3, figsize=(12, 6))

      axes[0].imshow( cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB))
      axes[0].set_title('Origin Image')
      axes[0].axis('off')

      axes[1].imshow( cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB))
      axes[1].set_title('Teacher Image')
      axes[1].axis('off')

      axes[2].imshow( cv2.cvtColor(teacher_img, cv2.COLOR_BGR2RGB))
      axes[2].set_title('Student Image')
      axes[2].axis('off')


      plt.show()
      """
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    else :
      student_img = self.augment(imageData)
      teacher_img = self.augment(imageData)

    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.image_list)
    



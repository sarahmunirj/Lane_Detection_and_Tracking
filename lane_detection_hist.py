import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import glob
import os
%load_ext autoreload
%autoreload 2
import importlib
import pickle
import time
from PIL import Image


frame_count = 0
Center_edit = []
def lane_detection_hist(image):
  
  global frame_count, Center_edit
  image = cv2.resize(image,(150,100))
  image_e = image[int(0.5*image.shape[0]):image.shape[0], :int(0.9*image.shape[1])]
  thresh = segregate_white_line(image_e)
  binary_warped,Minv,M=perspective_transform(thresh)
 

  nwindows = 3
  window_height = np.int(binary_warped.shape[0]/nwindows)
  # After creating a warped binary Image,
  # Take a histogram of the bottom half of the image
  y_high = binary_warped.shape[0] - window_height
  y_low = binary_warped.shape[1] 
  #print("y_h and y_l" + str(y_high) + str(y_low))
  histogram = np.sum(binary_warped[y_high:y_low,:], axis=0) #the 16 pixels down
  #print(" bin warped " + str(binary_warped[y_high:y_low,:]))
  #print(" bin warped2 " + str(binary_warped[:,y_high:y_low]))
  #histogram.append(np.sum(binary_warped, axis=0))
  right, left, center  = get_index_max(histogram)
  if (frame_count == 0 or frame_count ==1):
    Center_edit.append(center)
  else:
    Center_edit.append((center + Center_edit[frame_count-1] + Center_edit[frame_count-2])/3)
  error = (Center_edit[frame_count] - 67)/135
  
  frame_count = frame_count + 1
  return error
def get_index_max(hist):
  maxElementRight = np.amax(hist[int(0.5*hist.shape[0]):])
  maxElementLeft = np.amax(hist[:int(0.5*hist.shape[0])])
  indexRight = np.where(hist == maxElementRight)
  indexLeft = np.where(hist == maxElementLeft)
  if(len(indexRight[0])>1):
    indexRight = np.amax(indexRight[0])
    indexRight = int(indexRight)
  else:
    indexRight = int(indexRight[0])
  if(len(indexLeft[0])>1):
    indexLeft = np.amin(indexLeft[0])
    indexLeft = int(indexLeft)
  else:
    indexLeft = int(indexLeft[0])
  center_index = int(0.5*(indexRight+indexLeft))  
  return indexRight, indexLeft, center_index

def segregate_white_line(image,thresh=(200,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    l_binary=np.zeros_like(l_channel)
    l_binary[((l_channel>=200)&(l_channel<=255))]=1
    return l_binary

def perspective_transform(image, src = np.float32([[0,1],[0.185,0.278],[0.74,0.278],[1,1]]), dst = np.float32([[0.185,1],[0.185,0],[0.74,0],[0.74,1]]) ):
    #src=np.float32([[0,720],[450,550],[700,550],[1080,720]])
    #dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
    src = src * np.float32([image.shape[1], image.shape[0]])
    dst = dst * np.float32([image.shape[1], image.shape[0]])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv,M
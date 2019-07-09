# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:46:30 2019

@author: LiXiaoGang
"""

import os 
import random
import cv2 as cv
import numpy as np
import parameters
import encode_label


def read_dataset(path,labels,seed=0,balance=True):
    
    all_label_list = []
    Class_name = labels.Class_name
    num_classess = len(labels.Class_name)
    
    max_len = 0
    for i in range(num_classess):
        
        label_list_per_class = os.listdir(os.path.join(path,'Data',Class_name[i]))
        all_label_list.append(label_list_per_class)
        
        if len(label_list_per_class)> max_len:
            max_len = len(label_list_per_class)
    
    shuffle_all_label_list =[]
    
    # deal with class imbalance by copying samples.
    if  balance == True: 
        for i in range(len(all_label_list)):
            num = int(np.ceil(max_len/len(all_label_list[i])))
            
            z = []
            for j in range(num):
                z = z + all_label_list[i]
            shuffle_all_label_list = shuffle_all_label_list + z[0:max_len]
            
    # do not deal with class imbalance   
    if  balance == False:
        for i in range(len(all_label_list)):
            shuffle_all_label_list = shuffle_all_label_list + all_label_list[i]
        
    # shuffle list 
    random.seed(seed)
    random.shuffle(shuffle_all_label_list)
    return shuffle_all_label_list


def read_minibatch(i,batch_size,all_clips_name,mean_image):
    
    start = i*batch_size
    end = min(start+batch_size,len(all_clips_name))
    
    batch_clips_name = all_clips_name[start:end]
    clip_Y = encode_label.onehotencode(batch_clips_name)
    
    clip_X = np.zeros([batch_size,parameters.IN_DEPTH,parameters.IN_HEIGHT,
                       parameters.IN_WIDTH,parameters.IN_CHANNEL],dtype=np.float32)
    clip = np.zeros([parameters.IN_DEPTH,parameters.IN_HEIGHT,parameters.IN_WIDTH,
                     parameters.IN_CHANNEL],dtype=np.float32)
    
    for i in range(min(batch_size,end-start)):
        
        folder = batch_clips_name[i].split('_')[0]
        cap = cv.VideoCapture(os.path.join(os.path.dirname(os.getcwd()),'Data',folder,batch_clips_name[i]))
        
        for j in range(parameters.IN_DEPTH):
            ret,frame = cap.read()
            frame = cv.resize(frame,(parameters.IN_HEIGHT,parameters.IN_WIDTH)).astype(np.float32)
            frame = frame - mean_image    # remove mean image
            clip[j,:,:,:] = frame
        clip_X[i,:,:,:,:] = clip 
        
    return clip_Y,clip_X
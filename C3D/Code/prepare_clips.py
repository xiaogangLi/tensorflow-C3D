# -*- coding: utf-8 -*-

import os
import cv2 as cv
import parameters
import pandas as pd


path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

for j in range(len(labels.Class_name)):
    i = 0
    class_name = labels.Class_name[j]
    
    src_video_path = os.path.join(path,'Raw_Data',class_name)
    dst_clips_path = os.path.join(path,'Data',class_name)
    
    video_names = os.listdir(src_video_path)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    for name in video_names:
        
        print('Preprocessing:',name)
        cap = cv.VideoCapture(os.path.join(src_video_path,name))
        if cap.isOpened():
            
            num_frames = int(cap.get(7))
            if num_frames < parameters.IN_DEPTH:continue
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4) )
            
            
            frame_list = []
            for j in range(num_frames):
                ret, frame = cap.read()
                if ret == True:frame_list.append(frame)
            if len(frame_list) < parameters.IN_DEPTH:continue
            
            for j in range(int(len(frame_list)/parameters.IN_DEPTH)+1):
                
                start = j*parameters.STRIDE
                end = j*parameters.STRIDE + parameters.IN_DEPTH
                
                if (start>len(frame_list)) or (end > len(frame_list)):
                    clips = frame_list[-parameters.IN_DEPTH::]
                else:
                    clips = frame_list[j*parameters.STRIDE:j*parameters.STRIDE+parameters.IN_DEPTH]
    
                # write clips
                i += 1
                out = cv.VideoWriter(os.path.join(dst_clips_path,class_name+'_'+str(i)+'.avi'),fourcc, parameters.IN_DEPTH, (frame_width,frame_height))
                for k in range(parameters.IN_DEPTH):out.write(clips[k])
                       
out.release()                       
cap.release()    

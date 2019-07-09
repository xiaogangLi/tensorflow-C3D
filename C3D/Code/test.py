# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:06:48 2019

@author: LiXiaoGang
"""

import os
import sys
import cv2 as cv
import parameters
import numpy as np
import encode_label
import pandas as pd
import read_data as rd
import save_inference_model



def read_test_data(path):
    num_clips = int(sys.argv[1])    # 0 < num_clips <= the number of clips in test set.
    labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))    # load label.txt
    all_clips_name = rd.read_dataset(path,labels,seed=66,balance=False)    # test set
    mean_image = cv.imread(os.path.join(path,'Data','mean_image.jpg')).astype(np.float32)    # load mean image
    clip_Y,clip_X  = rd.read_minibatch(0,num_clips,all_clips_name,mean_image)
    return clip_Y,clip_X


def predict(clip_Y,clip_X):
    output = save_inference_model.inference(parameters.PB_MODEL_SAVE_PATH,clip_X)    # the output of softmax in model
    
    # Compute clip-level accuracy and predicted class name
    prediction = []
    acc_count = 0
    for one_output,one_clip_Y in zip(output,clip_Y):
        
        # Compute predicted class name
        pred_name = encode_label.onehotdecode(one_output)
        true_name = encode_label.onehotdecode(one_clip_Y)
        prediction.append({'Output':list(one_output),'Predicted_class_name':pred_name,'True_calss_name':true_name})
        
        # Compute clip-level accuracy
        if np.argmax(one_output) == np.argmax(one_clip_Y):
            acc_count += 1
    accuracy = (acc_count/len(prediction))
    return prediction,accuracy
               

def test_net():
    
    clip_Y,clip_X = read_test_data(parameters.path)
    prediction,accuracy = predict(clip_Y,clip_X)
    print('Clip_accuracy: %g' % accuracy)
    print(prediction)
    return prediction,accuracy
    
    
def main():
    return test_net()
    
    
if __name__ == '__main__':
    prediction,accuracy = main()    
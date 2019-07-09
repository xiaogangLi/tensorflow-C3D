# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:46:14 2019

@author: LiXiaoGang
https://tensorflow.google.cn/
https://github.com/tensorflow/serving
https://blog.csdn.net/thriving_fcl/article/details/75213361
https://www.cnblogs.com/mbcbyq-2137/p/10044837.html
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model

https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

https://www.jianshu.com/p/e5e36ffde809
https://www.jianshu.com/p/9221fbf52c55
"""

import os
import sys
import shutil
import c3d_net
import cv2 as cv
import parameters
import numpy as np
import pandas as pd
import read_data as rd
import tensorflow as tf
import save_inference_model


def load_clip_name(path):
    labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))    # load label.txt
    all_clips_name = rd.read_dataset(path,labels,seed=66,balance=True)    # train set
    mean_image = cv.imread(os.path.join(path,'Data','mean_image.jpg')).astype(np.float32)    # read mean image
    return all_clips_name,mean_image
        
 
def net_placeholder(batch_size=None):
    clip_X = tf.placeholder(tf.float32,shape=[batch_size,
                                              parameters.IN_DEPTH,
                                              parameters.IN_HEIGHT,
                                              parameters.IN_WIDTH,
                                              parameters.IN_CHANNEL],name='Input')
    clip_Y = tf.placeholder(tf.float32,shape=[batch_size,
                                              parameters.NUM_CLASSESS],name='Label')
    return clip_X,clip_Y
    

def net_loss(clip_Y,logits):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=clip_Y,logits=logits),name='loss')
    return loss
    

def training_net():
    all_clips_name,mean_image = load_clip_name(parameters.path)
    clip_X,clip_Y = net_placeholder(None)
    logits,Softmax_output = c3d_net.farward_c3d(clip_X)
    loss = net_loss(clip_Y,logits)
    train_step = tf.train.AdamOptimizer(parameters.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver()    # Saver
    with tf.Session() as sess:    # Launch the graph in a session.
        
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(parameters.CHECKPOINT_MODEL_SAVE_PATH, sess.graph)     
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        for i in range(parameters.TRAIN_STEPS):
            Y,X = rd.read_minibatch(i,parameters.BATCH_SIZE,all_clips_name,mean_image)
            feed_dict = {clip_X:X,clip_Y:Y}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            
            # ------------------------------- Save model --------------------------
            if i % 100 == 0: 
                
                # Way 1 : saving checkpoint model
                if sys.argv[1] == 'CHECKPOINT':                    
                    if os.path.exists(parameters.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                        shutil.rmtree(parameters.CHECKPOINT_MODEL_SAVE_PATH)
                    Saver.save(sess,os.path.join(parameters.CHECKPOINT_MODEL_SAVE_PATH,parameters.MODEL_NAME+str(i))) 
                        
                #  Way 2 : saving pb model       
                elif sys.argv[1] == 'PB':  
                    if os.path.exists(parameters.PB_MODEL_SAVE_PATH):
                        shutil.rmtree(parameters.PB_MODEL_SAVE_PATH)
                    save_inference_model.save_model(sess,parameters.PB_MODEL_SAVE_PATH,clip_X,Softmax_output) 
                else:
                    print('The argument is incorrect for the way saving model!')
                    sys.exit(0)
            print('===>Step %d: loss = %g ' % (i,loss_))
    

def main():
    training_net()
     
if __name__ == '__main__':
    main()
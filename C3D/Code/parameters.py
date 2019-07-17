# -*- coding: utf-8 -*-

import os
import pandas as pd


LEARNING_RATE = 0.0001
BATCH_SIZE = 30   
TRAIN_STEPS = 1000

IN_DEPTH = 16      
IN_HEIGHT = 128  
IN_WIDTH = 128 
IN_CHANNEL = 3 
STRIDE = 16

rate = 0.2
remove_mean_image = False

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

NUM_CLASSESS = len(labels.Class_name)
MODEL_NAME = 'model.ckpt-'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'Model','checkpoint')
PB_MODEL_SAVE_PATH = os.path.join(path,'Model','pb')

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:55:44 2019

@author: LiXiaoGang
"""

import os
import pandas as pd

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))



LEARNING_RATE = 0.0001
BATCH_SIZE = 2    # 视频片段数量（clips的数量）
TRAIN_STEPS = 1000

IN_DEPTH = 16       # clip 的深度，也即是clip视频片段的帧数
IN_HEIGHT = 128    # 帧的高度
IN_WIDTH = 128    # 帧的宽度
IN_CHANNEL = 3    # 3通道的RGB图像
STRIDE = 16     # 滑动窗取clips时的步长，clip_depth <= stride 表示取clip时，clip之间不存在重叠


NUM_CLASSESS = len(labels.Class_name)    # 视频类别数量
MODEL_NAME = 'model.ckpt-'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'Model','checkpoint')
PB_MODEL_SAVE_PATH = os.path.join(path,'Model','pb')
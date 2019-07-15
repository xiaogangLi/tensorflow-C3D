# -*- coding: utf-8 -*-

import parameters
import tensorflow as tf


def farward_c3d(clip_X):
    
    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(0)
    
    # -------------------------Conv 1 ---------------------------------------------
    Conv1_weights = tf.get_variable('Conv1_weights',shape=[3,3,3,parameters.IN_CHANNEL,64],initializer=weight_init)
    Conv1_biases = tf.get_variable('Conv1_biases',shape=[64],initializer=bias_init)
    Conv1 = tf.nn.conv3d(clip_X,Conv1_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv1')
    Conv1 = tf.nn.bias_add(Conv1,Conv1_biases)
    relu1 = tf.nn.relu(Conv1,name='relu1')
    # max_pool1
    max_pool1 = tf.nn.max_pool3d(relu1,ksize=[1,1,2,2,1],strides=[1,1,2,2,1],padding='SAME',name='Max_pool1')
    
    
    # ------------------------Conv 2 ----------------------------------------------
    Conv2_weights = tf.get_variable('Conv2_weights',shape=[3,3,3,64,128],initializer=weight_init)
    Conv2_biases = tf.get_variable('Conv2_biases',shape=[128],initializer=bias_init)
    Conv2 = tf.nn.conv3d(max_pool1,Conv2_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv2')
    Conv2 = tf.nn.bias_add(Conv2,Conv2_biases)
    relu2 = tf.nn.relu(Conv2,name='relu2')
    # max_pool2
    max_pool2 = tf.nn.max_pool3d(relu2,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME',name='Max_pool2')
    
    
    # ------------------------Conv 3 ----------------------------------------------
    Conv3_weights = tf.get_variable('Conv3_weights',shape=[3,3,3,128,256],initializer=weight_init)
    Conv3_biases = tf.get_variable('Conv3_biases',shape=[256],initializer=bias_init)
    Conv3 = tf.nn.conv3d(max_pool2,Conv3_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv3')
    Conv3 = tf.nn.bias_add(Conv3,Conv3_biases)
    relu3 = tf.nn.relu(Conv3,name='relu3')
    
    
    # ------------------------Conv 4 ----------------------------------------------
    Conv4_weights = tf.get_variable('Conv4_weights',shape=[3,3,3,256,256],initializer=weight_init)
    Conv4_biases = tf.get_variable('Conv4_biases',shape=[256],initializer=bias_init)
    Conv4 = tf.nn.conv3d(relu3,Conv4_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv4')
    Conv4 = tf.nn.bias_add(Conv4,Conv4_biases)
    relu4 = tf.nn.relu(Conv4,name='relu4')
    # max_pool3
    max_pool3 = tf.nn.max_pool3d(relu4,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME',name='Max_pool3')
    
    
    # ------------------------Conv 5 ----------------------------------------------
    Conv5_weights = tf.get_variable('Conv5_weights',shape=[3,3,3,256,512],initializer=weight_init)
    Conv5_biases = tf.get_variable('Conv5_biases',shape=[512],initializer=bias_init)
    Conv5 = tf.nn.conv3d(max_pool3,Conv5_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv5')
    Conv5 = tf.nn.bias_add(Conv5,Conv5_biases)
    relu5 = tf.nn.relu(Conv5,name='relu5')
    
    
    # ------------------------Conv 6 ----------------------------------------------
    Conv6_weights = tf.get_variable('Conv6_weights',shape=[3,3,3,512,512],initializer=weight_init)
    Conv6_biases = tf.get_variable('Conv6_biases',shape=[512],initializer=bias_init)
    Conv6 = tf.nn.conv3d(relu5,Conv6_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv6')
    Conv6 = tf.nn.bias_add(Conv6,Conv6_biases)
    relu6 = tf.nn.relu(Conv6,name='relu6')
    # max_pool4
    max_pool4 = tf.nn.max_pool3d(relu6,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME',name='Max_pool4')
    
    # ------------------------Conv 7 ----------------------------------------------
    Conv7_weights = tf.get_variable('Conv7_weights',shape=[3,3,3,512,512],initializer=weight_init)
    Conv7_biases = tf.get_variable('Conv7_biases',shape=[512],initializer=bias_init)
    Conv7 = tf.nn.conv3d(max_pool4,Conv7_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv7')
    Conv7 = tf.nn.bias_add(Conv7,Conv7_biases)
    relu7 = tf.nn.relu(Conv7,name='relu7')
    
    
    # ------------------------Conv 8 ----------------------------------------------
    Conv8_weights = tf.get_variable('Conv8_weights',shape=[3,3,3,512,512],initializer=weight_init)
    Conv8_biases = tf.get_variable('Conv8_biases',shape=[512],initializer=bias_init)
    Conv8 = tf.nn.conv3d(relu7,Conv8_weights,strides=[1,1,1,1,1],padding='SAME',name='Conv8')
    Conv8 = tf.nn.bias_add(Conv8,Conv8_biases)
    relu8 = tf.nn.relu(Conv8,name='relu8')
    # max_pool5
    max_pool5 = tf.nn.max_pool3d(relu8,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME',name='Max_pool5')
    
    
    # ------------------------Flatten ----------------------------------------------
    reshape_vector = tf.layers.flatten(max_pool5,name='Flatten')
    reshape_vector_dimens = reshape_vector.get_shape().as_list()[-1]
    
    # ------------------------Dense 1 ---------------------------------------------
    D1_weights = tf.get_variable('D1_weights',shape=[reshape_vector_dimens,1024],initializer=weight_init)
    D1_biases = tf.get_variable('D1_biases',shape=[1024],initializer=bias_init)
    D1 =tf.nn.relu(tf.add(tf.matmul(reshape_vector,D1_weights),D1_biases),name='D1_relu')
    
    
    # ------------------------Dense 2 ---------------------------------------------
    D2_weights = tf.get_variable('D2_weights',shape=[1024,1024],initializer=weight_init)
    D2_biases = tf.get_variable('D2_biases',shape=[1024],initializer=bias_init)
    D2 = tf.nn.relu(tf.add(tf.matmul(D1,D2_weights),D2_biases),name='D2_relu')
    
    
    # ------------------------Softmax ---------------------------------------------
    Softmax_weights = tf.get_variable('Softmax_weights',shape=[1024,parameters.NUM_CLASSESS],initializer=weight_init)
    Softmax_biases = tf.get_variable('Softmax_biases',shape=[parameters.NUM_CLASSESS],initializer=bias_init)
    logits = tf.add(tf.matmul(D2,Softmax_weights),Softmax_biases)
    Softmax_output = tf.nn.softmax(logits,name='Softmax_output')
    
    return logits,Softmax_output

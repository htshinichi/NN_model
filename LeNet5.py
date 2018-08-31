# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:31:04 2018

@author: htshinichi
"""
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
DATA_PATH="C:/Users/htshinichi/Desktop/DL-TF/MNIST_TEST/MnistTest/DATA" 
mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

BATCH_SIZE = 100
#原始输入尺寸和深度
INPUT_SIZE = 32
NUM_CHANNELS = 1
#原始数据类别
NUM_LABELS = 10

#卷积层1尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 6

#卷积层2尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 16

#全连接层1节点数
FC1_SIZE = 120
#全连接层2节点数
FC2_SIZE = 84
#全连接层3节点数
FC3_SIZE = 10

###--------------------------------------------------定义神经网络前向传播过程---------------------------------------------###
###inference(input,train,regularizer)
###input:[batch,in_height,in_width,in_channels]=[训练一个batch的图片数量,图片高度,图片宽度,图像通道数]
###train:用于区分训练过程和测试过程，因为dropout只在训练时使用
###regularizer:正则化

def inference(input,train,regularizer):
    
    ##第一层：卷积层1，卷积核尺寸为5x5，深度为6，不使用全0填充，步长为1
    ##尺寸变化：32x32x1 -> 28x28x6
    ##下一层节点矩阵有28x28x6=4704个节点，每个节点和当前层5x5=25个节点相连
    ##本层卷积层共有4704x(25+1)=122304个连接(1为偏置)
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        ##--------------------卷积函数tf.nn.conv2d()------------------------###
        ##tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        ##input=[batch,in_height,in_width,in_channels]=[训练一个batch的图片数量,图片高度,图片宽度,图像通道数]
        ##filter=[filter_height,filter_width,in_channels,out_channels]=[卷积核高,卷积核宽,图像通道数,卷积核个数]
        ##strides:卷积时在图像每一维的步长,一维向量,长度为4
        ##padding:string类型,只能是"SAME"(全0填充)或者"VALID"(非全0填充)
        ##use_cudnn_on_gpu:bool类型,是否使用cudnn加速,默认为true
        ##name:给返回的tensor命名
        conv1 = tf.nn.conv2d(input,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    
    ##第二层：池化层1，过滤器尺寸为2x2，使用全0填充，步长为2
    ##尺寸变化：28x28x6 -> 14x14x6
    ##下一层节点矩阵有14x14x6=1176个节点，每个节点和当前层2x2=4个节点相连
    ##本层池化层共有1176x(4+1)=5880个连接
    with tf.name_scope('layer2-pool1'):
        ##------------------池化函数tf.nn.max_pool()------------------------###
        ##tf.nn.max_pool(value, ksize, strides, padding, name=None)
        ##value=[batch,in_height,in_width,in_channels],与卷积函数中input相同
        ##ksize=[1,height,width,1],池化窗口的大小(由于我们不想在batch和channels上池化)，一维向量，长度为4
        ##strides:池化时窗口在每一维度的步长，一维向量，长度为4
        ##padding:string类型,只能是"SAME"(全0填充)或者"VALID"(非全0填充)
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        
    ##第三层：卷积层2，卷积核尺寸为5x5，深度为16，不使用全0填充，步长为1 
    ##尺寸变化：14x14x6 -> 10x10x16
    ##下一层节点矩阵有10x10x16=1600个节点，每个节点和当前层5x5=25个节点相连
    ##本层卷积层共有1600x(25+1)=41600个连接(1为偏置)
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    
    
    ##第四层：池化层2，过滤器尺寸为2x2，使用全0填充，步长为2
    ##尺寸变化：10x10x16 -> 5x5x6
    ##下一层节点矩阵有5x5x16=400个节点，每个节点和当前层2x2=4个节点相连
    ##本层池化层共有400x(4+1)=2000个连接
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        
    ##将第四层的输出转化为第五层全连接层的输入格式
    ##第四层输出：5x5x16，第五层输入：向量
    ##(batch,in_height,in_width,in_channels) -> (batch,in_height x in_width x in_channels)
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])
    
    
    ##第五层：全连接层1，node=5x5x16=400
    ##尺寸变化：batch x 400 -> batch x 120
    ##下一层节点个数为120个，每个节点与当前层400个节点相连
    ##本层全连接层共有120x(400+1)=48120个参数
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight",[nodes,FC1_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[FC1_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
            
    
    ##第六层：全连接层2
    ##尺寸变化：batch x 120 -> batch x 84
    ##下一层节点个数为84个，每个节点与当前层120个节点相连
    ##本层全连接层共有84x(120+1)=10164个参数
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight",[FC1_SIZE,FC2_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias",[FC2_SIZE],initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
            
            
    ##第七层：全连接层3
    ##尺寸变化：batch x 84 -> batch x 10
    ##下一层节点个数为10个，每个节点与当前层84个节点相连
    ##本层全连接层共有10x(84+1)=850个参数
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable("weight",[FC2_SIZE,FC3_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias",[FC3_SIZE],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases
        
    return logit

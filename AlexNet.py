# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:31:04 2018
AlexNet前向传播模型
参考并学习了下面的博客：
https://blog.csdn.net/u011974639/article/details/76146822#
https://www.cnblogs.com/Yu-FeiFei/p/6800519.html
###############################################################################
  AlexNet模型（2012）是CNN模型复兴的开山之作，论文中介绍了完整的CNN架构模型。
-------------------------------------------------------------------------------  
  在网络结构上: 1.成功使用了ReLU激活函数，验证其效果在较深的网络要由于sigmoid。
               2.应用了(重叠)最大池化层，提出让步长小于池化核尺寸，这样池化层的输出
                 之间会有重叠和覆盖，提升了特征的丰富性。
               3.使用LRN层，对局部神经元的活动创建竞争机制，使其中响应比较大的值变得
                 相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
  在训练过程上: 1.数据增强(对原始图像随机截取输入图片固定尺寸以及对图像做水平翻转来扩
                 大数据集)。
               2.使用dropout技术增强模型的鲁棒性和泛化能力。
               3.测试时，对测试图片做数据增强，同时在softmax层去均值再输出。
               4.带动量的权值衰减。
               5.学习率在验证集的错误率不提高时下降10倍。
               6.使用GPU加速，并使用两块GPU并行计算（下面的程序中并未分成两块）
  其他观点：LRN层(Local Response Normalization，局部响应归一化)有助于增强模型泛化
           能力。
-------------------------------------------------------------------------------
###############################################################################

@author: htshinichi
"""
import tensorflow as tf 

#原始输入尺寸和深度
INPUT_SIZE = 227
NUM_CHANNELS = 3
#卷积层1尺寸和深度
CONV1_SIZE = 11
CONV1_DEEP = 96
#卷积层2尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 256
#卷积层3尺寸和深度
CONV3_SIZE = 3
CONV3_DEEP = 384
#卷积层4尺寸和深度
CONV4_SIZE = 3
CONV4_DEEP = 384
#卷积层5尺寸和深度
CONV5_SIZE = 3
CONV5_DEEP = 256
#全连接层1节点数
FC1_SIZE = 4096
#全连接层2节点数
FC2_SIZE = 4096
#全连接层(softmax层)
FC3_SIZE = 1000


###--------------------------------------------------定义神经网络前向传播过程---------------------------------------------###
###inference(input,train,regularizer)
###input:[batch,in_height,in_width,in_channels]=[训练一个batch的图片数量,图片高度,图片宽度,图像通道数]
###train:用于区分训练过程和测试过程，因为dropout只在训练时使用
###regularizer:正则化

def inference(input,train,regularizer):
    
    ##第一层：卷积层1，卷积核尺寸为11x11x3，深度为96，不使用全0填充(pad=0)，步长为4
    ##(227-11)/4 + 1 = 55
    ##尺寸变化：227x227x3 -> 55x55x96
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weights", [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        ##--------------------卷积函数tf.nn.conv2d()------------------------###
        ##tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        ##input=[batch,in_height,in_width,in_channels]=[训练一个batch的图片数量,图片高度,图片宽度,图像通道数]
        ##filter=[filter_height,filter_width,in_channels,out_channels]=[卷积核高,卷积核宽,图像通道数,卷积核个数]
        ##strides:卷积时在图像每一维的步长,一维向量,长度为4
        ##padding:string类型,只能是"SAME"(全0填充)或者"VALID"(非全0填充)
        ##use_cudnn_on_gpu:bool类型,是否使用cudnn加速,默认为true
        ##name:给返回的tensor命名
        conv1 = tf.nn.conv2d(input,conv1_weights,strides=[1,4,4,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    
    ##第一层：池化层1，过滤器尺寸为3x3，不使用全0填充，步长为2
    ##(55-3)/2 + 1 = 27
    ##尺寸变化：55x55x96 -> 27x27x96
    with tf.name_scope('layer1-pool1'):
        ##------------------池化函数tf.nn.max_pool()------------------------###
        ##tf.nn.max_pool(value, ksize, strides, padding, name=None)
        ##value=[batch,in_height,in_width,in_channels],与卷积函数中input相同
        ##ksize=[1,height,width,1],池化窗口的大小(由于我们不想在batch和channels上池化)，一维向量，长度为4
        ##strides:池化时窗口在每一维度的步长，一维向量，长度为4
        ##padding:string类型,只能是"SAME"(全0填充)或者"VALID"(非全0填充)
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

##############################################################################################################     
    ##第二层：卷积层2，卷积核尺寸为5x5x96，深度为256，使用全0填充(pad=2)，步长为1
    ##(27+2x2-5)/1 + 1 = 27
    ##尺寸变化：27x27x96 -> 27x27x256
    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weights", [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))    
    
    ##第二层：池化层2，过滤器尺寸为3x3，不使用全0填充，步长为2
    ##(27-3)/2 +1 = 13
    ##尺寸变化：27x27x256 -> 13x13x256
    with tf.name_scope('layer2-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
##############################################################################################################        
    ##第三层：卷积层3，卷积核尺寸为3x3x256，深度为384，使用全0填充(pad=1)，步长为1
    ##(13+2x1-3)/1 + 1 = 13
    ##尺寸变化：13x13x256 -> 13x13x384
    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weights", [CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))

##############################################################################################################        
    ##第四层：卷积层4，卷积核尺寸为3x3x384，深度为384，使用全0填充(pad=1)，步长为1
    ##(13+2x1-3)/1 + 1 = 13
    ##尺寸变化：13x13x384 -> 13x13x384
    with tf.variable_scope('layer4-conv4'):
        conv4_weights = tf.get_variable("weights", [CONV4_SIZE,CONV4_SIZE,CONV3_DEEP,CONV4_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))     
        
##############################################################################################################        
    ##第五层：卷积层5，卷积核尺寸为3x3x384，深度为256，使用全0填充(pad=1)，步长为1
    ##(13+2x1-3)/1 + 1 = 13
    ##尺寸变化：13x13x384 -> 13x13x256
    with tf.variable_scope('layer5-conv5'):
        conv5_weights = tf.get_variable("weights", [CONV5_SIZE,CONV5_SIZE,CONV4_DEEP,CONV5_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4,conv5_weights,strides=[1,1,1,1],padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))   

    ##第五层：池化层3，过滤器尺寸为3x3，不使用全0填充，步长为2
    ##(13-3)/2 +1 = 6
    ##尺寸变化：13x13x256 -> 6x6x256
    with tf.name_scope('layer5-pool3'):
        pool3 = tf.nn.max_pool(relu5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

        
    ##将第五层的输出转化为第六层全连接层的输入格式
    ##第五层输出：6x6x256，第六层输入：向量
    ##(batch,in_height,in_width,in_channels) -> (batch,in_height x in_width x in_channels)
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool3,[-1,nodes])
    
##############################################################################################################    
    ##第六层：全连接层1，node=6x6x256=9216
    ##尺寸变化：batch x 9216 -> batch x 4096
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable("weights",[nodes,FC1_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[FC1_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
            
##############################################################################################################    
    ##第七层：全连接层2，node=4096
    ##尺寸变化：batch x 4096 -> batch x 4096
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable("weights",[FC1_SIZE,FC2_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias",[FC2_SIZE],initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
            
##############################################################################################################            
    ##第八层：全连接层3(输出层，softmax层)
    ##尺寸变化：batch x 4096 -> batch x 1000
    ##训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题
    with tf.variable_scope('layer8-fc3'):
        fc3_weights = tf.get_variable("weights",[FC2_SIZE,FC3_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias",[FC3_SIZE],initializer=tf.constant_initializer(0.1))
        fc3 = tf.matmul(fc2,fc3_weights) + fc3_biases
        logit = tf.nn.softmax(fc3)
        
    return logit

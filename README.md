# Attention-Mixer based Transformer Network for Time Series Processing![Uploading image.png…]()
Master Thesis of Junkun Lu 

## 一. 模型介绍
![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/%E6%A8%A1%E5%9E%8B.png)
在本论文中, 我们搭建了一个基于混合注意力机制的Transformer优化模型, 该模型主要由三部分组成:
  1. 基于Inception Module的原理使用Local-Attention, LogSparse-Attention以及ProbSparse-Attention的融合多头注意力机制代替Transformer原有的多头注意力机制;
  2. 搭建一个组合型预测器实现从不同维度学习表征并降低维度;
  3. 在损失函数中引入平滑正则化项, 降低模型在预测中的振荡.
 
## 二. 数据集介绍
在本论文中, 我们分别从分类(人类动作识别任务)与回归(剩余寿命预测任务)两个方面分别对搭建的优化模型进行验证.
  1.针对于分类任务, 我们采用了UCI-HAR数据集; 
  2.针对于回归任务, 我们采用了CMAPSS数据集, 在数据的处理过程中,以120为拐点对数据进行处理. 将125之前的数据定义为稳定运行过程,125之后的数据定义为衰退过程.

## 三. 实验结果
在本论文中, 我们分别针对两个数据集进行消融实验, 验证了融合型自注意力机制的优越性以及在预测过程过平滑正则化项的有效性.
  ### 1. UCI-HAR
    在UCI-HAR数据集中, 我们将混合型自注意力机制
  

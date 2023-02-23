# Attention-Mixer based Transformer Network for Time Series Processing
Master Thesis of Junkun Lu 

## 一. 模型介绍
![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/%E6%A8%A1%E5%9E%8B.png)
在本论文中, 我们搭建了一个基于混合注意力机制的Transformer优化模型, 该模型主要由三部分组成:
  1. 基于Inception Module的原理使用Local-Attention, LogSparse-Attention以及ProbSparse-Attention的融合多头注意力机制代替Transformer原有的多头注意力机制;
  2. 搭建一个组合型预测器实现从不同维度学习表征并降低维度;
  3. 在损失函数中引入平滑正则化项, 降低模型在预测中的振荡.
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)
 
## 二. 数据集介绍
在本论文中, 我们分别从分类(人类动作识别任务)与回归(剩余寿命预测任务)两个方面分别对搭建的优化模型进行验证.
  1.针对于分类任务, 我们采用了UCI-HAR数据集; 
  2.针对于回归任务, 我们采用了CMAPSS数据集, 在数据的处理过程中,以120为拐点对数据进行处理. 将120之前的数据定义为稳定运行过程,120之后的数据定义为衰退过程.
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/CMAPSS%E5%A4%84%E7%90%86.png)
  
## 三. 实验结果
在本论文中, 我们分别针对两个数据集进行消融实验, 验证了融合型自注意力机制的优越性以及在预测过程过平滑正则化项的有效性.
  
### 1. UCI-HAR
  a) 在UCI-HAR数据集中, 我们将混合型自注意力机制与DeepConvLSTM进行对比,得到如下结果:
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/UCI_HAR%E5%AF%B9%E6%AF%94%E8%AF%95%E9%AA%8C.png)
  b) 通过消融实验, 我们最终得出, 三种优化自注意力混合机制的Transformer优化模型在处理时间序列相关的人类动作识别任务上有着最优的结果:
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/UCI_HAR%E6%B6%88%E8%9E%8D%E5%AE%9E%E9%AA%8C.png)
  
### 2. CMAPSS
  a) 在CMAPSS数据集中, 我们针对四个子数据集, 将其结果分别与多种流行的深度学习模型进行对比,得到了不错的结果:
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/CMAPSS-%E5%AF%B9%E6%AF%94%E8%AF%95%E9%AA%8C.png)
  b) 通过消融实验, 我们对比了不同优化自注意力机制及其融合机制与标准的自注意力机制在CMAPSS数据集上的预测结果:
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/%E6%B6%88%E8%9E%8D%E5%AE%9E%E9%AA%8C-CMAPSS.png)
  c) 通过消融实验,我们分别对比了使用平滑正则化项与不使用平滑正则化项训练的结果, 最终发现,使用平滑正则化项,结果会产生更小的置信区间, 增加了模型预测的鲁棒性:
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/CMAPSS-%E5%B9%B3%E6%BB%91%E6%AD%A3%E5%88%99%E5%8C%96.png)
  ![Image text](https://github.com/Junkun-Lu/Attention-Mixer-based-Transformer-Network-for-Time-Series-Processing/blob/main/picture/%E9%A2%84%E6%B5%8B%E8%BF%87%E7%A8%8B.png)
  

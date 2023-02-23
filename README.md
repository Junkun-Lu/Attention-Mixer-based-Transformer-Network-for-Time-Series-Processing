# Inception-Attention-Network
Master Thesis of Junkun Lu 

在本论文中, 我们搭建了一个基于混合注意力机制的Transformer优化模型, 该模型主要由三部分组成:
  1. 基于Inception Module的原理使用Local-Attention, LogSparse-Attention以及ProbSparse-Attention的融合多头注意力机制代替Transformer原有的多头注意力机制;
  2. 搭建一个组合型预测器实现从不同维度学习表征并降低维度;
  3. 在损失函数中引入平滑正则化项, 降低模型在预测中的振荡.
 

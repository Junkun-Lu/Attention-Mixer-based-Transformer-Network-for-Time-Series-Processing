Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)



Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine ?i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

数据集由多个多变量时间序列组成。每个数据集被进一步分为训练和测试子集。每个时间序列都来自不同的发动机，也就是说，数据可以被认为是来自同一类型的发动机车队。每台发动机开始时都有不同程度的初始磨损和制造变化，这对用户来说是未知的。这种磨损和变化被认为是正常的，也就是说，它不被认为是一种故障状况。有三种操作设置对发动机性能有很大影响。这些设置也包括在数据中。数据被传感器的噪声所污染。

发动机在每个时间序列开始时运行正常，并在序列中的某个时刻出现了故障。在训练集中，故障的幅度越来越大，直到系统失效。在测试组中，时间序列在系统故障之前的某个时间结束。比赛的目的是预测测试集中故障前的剩余运行周期数，即最后一个周期后发动机将继续运行的运行周期数。还提供了一个测试数据的真实剩余使用寿命（RUL）值的向量。

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, 揇amage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation? in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

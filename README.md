## 流量矩阵预测相关代码

Abilene、CERNET和GEANT三个文件夹下，是针对三个拓扑的流量矩阵预测代码和结果文件。大概的结构如下：

* model_xxx：pytorch pkl文件
* xxx.csv：预测结果、分析结果（根据文件名可以判断）
* overflow_xxx：之前的一个想法，暂时没有实现，请忽略
* draw_fig.py：绘图分析脚本
* 方法.py：不同预测方法的代码
* hybrid.py：对SOTE预测结果进行进一步分析，获得MLU bias MAE等统计性结果
* OSPF_weight.py：将预测得到的TM转化为OSPF/SOTE脚本需要的输入格式，并进一步对预测结果进行分析，获得MLU bias MAE等统计性结果
* SDN_split_ratio.py：在MCF C++代码的基础上，进一步进行Prediction based TE（基于MCF C++的分流结果，路由真实的TM）



TM_result文件夹是预测结果以及路由策略结果。其中路由策略结果保存了OSPF（OSPF_weight文件夹下）和MCF（split_ratio文件夹下）两种场景。SOTE场景不需要保存。



statistical_analysis文件夹主要是进行实验结果的统计分析，形成论文中的表格。



>注意，TM预测的预测步长为10（早期设定的，后来在单点预测中改为了50）。同时，没有尝试较大的 epoch。EKM方法的epoch为50-100，其他收敛较快的方法的epoch为20-50。后续可以进一步进行调整。

## update on 2020.10.12

1. remove intermediate result files and irrelevant files to eliminate privacy 
2. modify some absolute paths to relative paths to facilitate reproduction
3. generate requirements.txt

If there is any problem, please submit a issue or email to zhao-yf20@mails.tsinghua.edu.cn.

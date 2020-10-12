import csv
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import os

'''
暂定结果图分为若干种情况：
    · 直接预测 TM：LSTM、GRU、TCN、LRCN
    · 预测每个 OD flow，同时对比 LSTM、GRU：DBN-OD、LSTM-OD、GRU-OD、LSTM-EKM-OD、LSTM、GRU

    · KEC 获得预测准确度和预测时间权衡
        · LSTM、LSTM-OD、LSTM-KEC
        · LSTM、LSTM-EKM-OD、LSTM-EKM-KEC

    · 如上述，每个拓扑总共 4 张图
'''


# 绘制 RMSE 结果
def draw_TM_RMSE(file_name, topology):
    df = pd.read_csv(file_name)

    # 直接预测 TM
    labels = ["LSTM", "GRU", "CNN_LSTM", "TCN"]
    color_dict = {"LSTM": "red", "GRU": "blue", "CNN_LSTM": "orange", "TCN": "green"}
    style = {"LSTM": '-', "GRU": '--', "CNN_LSTM": "-.", "TCN": ":"}
    plt.xlabel("RMSE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    width = 1.5
    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/RMSE_TM_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # 预测每个 OD flow，同时对比 LSTM、GRU
    labels = ["LSTM", "GRU", "DBN", "LSTM_OD_pair", "GRU_OD_pair", "LSTM-EKM_OD_pair", "GRU-EKM_OD_pair"]

    color_dict = {"LSTM": "deepskyblue", "GRU": "blue", "DBN": "green", "LSTM_OD_pair": "orange",
                  "GRU_OD_pair": "coral", "LSTM-EKM_OD_pair": "red", "GRU-EKM_OD_pair": "purple"}

    style = {"LSTM":'-', "GRU":'--', "DBN":"-.", "LSTM_OD_pair":":", "GRU_OD_pair":"-.",
             "LSTM-EKM_OD_pair":"--", "GRU-EKM_OD_pair":"-"}

    plt.xlabel("RMSE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "LSTM_OD_pair" or label == "GRU_OD_pair" or \
                        label == "LSTM-EKM_OD_pair" or label == "GRU-EKM_OD_pair":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "DBN":
            labels[i] = "DBN-OD"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/RMSE_OD_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # LSTM_EKM_KEC
    labels = ["LSTM", "LSTM_OD_pair", "LSTM-EKM_OD_pair", "LSTM_EKM_KEC_25", "LSTM_EKM_KEC_53", "LSTM_EKM_KEC_106",
              "LSTM_EKM_KEC_159"]

    color_dict = {"LSTM": "deepskyblue", "LSTM-EKM_OD_pair": "blue", "LSTM_EKM_KEC_25": "orange",
                  "LSTM_EKM_KEC_53": "green",
                  "LSTM_EKM_KEC_106": "purple", "LSTM_EKM_KEC_159": "coral", "LSTM_OD_pair": "red"}

    style = {"LSTM": '-', "LSTM-EKM_OD_pair": '--', "LSTM_EKM_KEC_25": ":", "LSTM_EKM_KEC_53": "-",
             "LSTM_EKM_KEC_106": "--", "LSTM_EKM_KEC_159": "-.", "LSTM_OD_pair": "-."}

    plt.xlabel("RMSE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "LSTM_EKM_KEC_53" or label == "LSTM_EKM_KEC_106" or label == "LSTM_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "LSTM_EKM_KEC_25":
            labels[i] = "LSTM-EKM-KEC-5%"
        if labels[i] == "LSTM_EKM_KEC_53":
            labels[i] = "LSTM-EKM-KEC-10%"
        if labels[i] == "LSTM_EKM_KEC_106":
            labels[i] = "LSTM-EKM-KEC-20%"
        if labels[i] == "LSTM_EKM_KEC_159":
            labels[i] = "LSTM-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/RMSE_LSTM-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()



    # GRU_EKM_KEC
    labels = ["GRU", "GRU_OD_pair", "GRU-EKM_OD_pair", "GRU_EKM_KEC_25", "GRU_EKM_KEC_53", "GRU_EKM_KEC_106",
              "GRU_EKM_KEC_159"]

    color_dict = {"GRU": "deepskyblue", "GRU-EKM_OD_pair": "blue", "GRU_EKM_KEC_25": "orange",
                  "GRU_EKM_KEC_53": "green",
                  "GRU_EKM_KEC_106": "purple", "GRU_EKM_KEC_159": "coral", "GRU_OD_pair": "red"}

    style = {"GRU": '-', "GRU-EKM_OD_pair": '--', "GRU_EKM_KEC_25": ":", "GRU_EKM_KEC_53": "-",
             "GRU_EKM_KEC_106": "--", "GRU_EKM_KEC_159": "-.", "GRU_OD_pair": "-."}

    plt.xlabel("RMSE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "GRU_EKM_KEC_53" or label == "GRU_EKM_KEC_106" or label == "GRU_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "GRU_EKM_KEC_25":
            labels[i] = "GRU-EKM-KEC-5%"
        if labels[i] == "GRU_EKM_KEC_53":
            labels[i] = "GRU-EKM-KEC-10%"
        if labels[i] == "GRU_EKM_KEC_106":
            labels[i] = "GRU-EKM-KEC-20%"
        if labels[i] == "GRU_EKM_KEC_159":
            labels[i] = "GRU-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/RMSE_GRU-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


# 绘制 MAE 结果
def draw_TM_MAE(file_name, topology):
    df = pd.read_csv(file_name)

    # 直接预测 TM
    labels = ["LSTM", "GRU", "CNN_LSTM", "TCN"]
    color_dict = {"LSTM": "red", "GRU": "blue", "CNN_LSTM": "orange", "TCN": "green"}
    style = {"LSTM": '-', "GRU": '--', "CNN_LSTM": "-.", "TCN": ":"}
    plt.xlabel("MAE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    width = 1.5
    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/MAE_TM_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # 预测每个 OD flow，同时对比 LSTM、GRU
    labels = ["LSTM", "GRU", "DBN", "LSTM_OD_pair", "GRU_OD_pair", "LSTM-EKM_OD_pair", "GRU-EKM_OD_pair"]

    color_dict = {"LSTM": "deepskyblue", "GRU": "blue", "DBN": "green", "LSTM_OD_pair": "orange",
                  "GRU_OD_pair": "coral", "LSTM-EKM_OD_pair": "red", "GRU-EKM_OD_pair": "purple"}

    style = {"LSTM":'-', "GRU":'--', "DBN":"-.", "LSTM_OD_pair":":", "GRU_OD_pair":"-.",
             "LSTM-EKM_OD_pair":"--", "GRU-EKM_OD_pair":"-"}

    plt.xlabel("MAE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "LSTM_OD_pair" or label == "GRU_OD_pair" or \
                        label == "LSTM-EKM_OD_pair" or label == "GRU-EKM_OD_pair":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "DBN":
            labels[i] = "DBN-OD"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/MAE_OD_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # LSTM_EKM_KEC
    labels = ["LSTM", "LSTM_OD_pair", "LSTM-EKM_OD_pair", "LSTM_EKM_KEC_25", "LSTM_EKM_KEC_53", "LSTM_EKM_KEC_106",
              "LSTM_EKM_KEC_159"]

    color_dict = {"LSTM": "deepskyblue", "LSTM-EKM_OD_pair": "blue", "LSTM_EKM_KEC_25": "orange",
                  "LSTM_EKM_KEC_53": "green",
                  "LSTM_EKM_KEC_106": "purple", "LSTM_EKM_KEC_159": "coral", "LSTM_OD_pair": "red"}

    style = {"LSTM": '-', "LSTM-EKM_OD_pair": '--', "LSTM_EKM_KEC_25": ":", "LSTM_EKM_KEC_53": "-",
             "LSTM_EKM_KEC_106": "--", "LSTM_EKM_KEC_159": "-.", "LSTM_OD_pair": "-."}

    plt.xlabel("MAE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "LSTM_EKM_KEC_53" or label == "LSTM_EKM_KEC_106" or label == "LSTM_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "LSTM_EKM_KEC_25":
            labels[i] = "LSTM-EKM-KEC-5%"
        if labels[i] == "LSTM_EKM_KEC_53":
            labels[i] = "LSTM-EKM-KEC-10%"
        if labels[i] == "LSTM_EKM_KEC_106":
            labels[i] = "LSTM-EKM-KEC-20%"
        if labels[i] == "LSTM_EKM_KEC_159":
            labels[i] = "LSTM-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/MAE_LSTM-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


    # GRU_EKM_KEC
    labels = ["GRU", "GRU_OD_pair", "GRU-EKM_OD_pair", "GRU_EKM_KEC_25", "GRU_EKM_KEC_53", "GRU_EKM_KEC_106",
              "GRU_EKM_KEC_159"]

    color_dict = {"GRU": "deepskyblue", "GRU-EKM_OD_pair": "blue", "GRU_EKM_KEC_25": "orange",
                  "GRU_EKM_KEC_53": "green",
                  "GRU_EKM_KEC_106": "purple", "GRU_EKM_KEC_159": "coral", "GRU_OD_pair": "red"}

    style = {"GRU": '-', "GRU-EKM_OD_pair": '--', "GRU_EKM_KEC_25": ":", "GRU_EKM_KEC_53": "-",
             "GRU_EKM_KEC_106": "--", "GRU_EKM_KEC_159": "-.", "GRU_OD_pair": "-."}

    plt.xlabel("MAE(Mbps)")
    plt.ylabel("CDF")
    plt.ylim((0, 1))
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    for label in labels:
        if label == "GRU_EKM_KEC_53" or label == "GRU_EKM_KEC_106" or label == "GRU_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "GRU_EKM_KEC_25":
            labels[i] = "GRU-EKM-KEC-5%"
        if labels[i] == "GRU_EKM_KEC_53":
            labels[i] = "GRU-EKM-KEC-10%"
        if labels[i] == "GRU_EKM_KEC_106":
            labels[i] = "GRU-EKM-KEC-20%"
        if labels[i] == "GRU_EKM_KEC_159":
            labels[i] = "GRU-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/MAE_GRU-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


# 绘制 OD flow prediction bias 散点图
# 纵坐标为逆归一化后的 prediction bias
# 横坐标为 OD flow 序列，从大到小排列，1 代表平均流量最大的 OD flow，以此类推
def draw_OD_bias(file_name, topology):
    df = pd.read_csv(file_name)

    # 直接预测 TM
    labels = ["LSTM", "GRU", "CNN_LSTM", "TCN"]
    color_dict = {"LSTM": "red", "GRU": "blue", "CNN_LSTM": "orange", "TCN": "green"}
    style = {"LSTM":'o', "GRU":'x', "CNN_LSTM":"*", "TCN":"+"}
    plt.xlabel("Flow ID, From Largest to Smallest in Mean")
    plt.xlim((0, 529))
    plt.ylabel("Bias(Mbps)")

    # 科学计数法显示横坐标
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)


    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)

        plt.scatter(range(data_size), data, marker=style[label], color=color_dict[label], s=20)

    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/Bias_TM_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


    # 预测每个 OD flow，同时对比 LSTM、GRU
    labels = ["LSTM", "GRU", "DBN", "LSTM_OD_pair", "GRU_OD_pair", "LSTM-EKM_OD_pair", "GRU-EKM_OD_pair"]

    color_dict = {"LSTM": "red", "GRU": "blue", "DBN": "orange", "LSTM_OD_pair": "green",
                  "GRU_OD_pair": "purple", "LSTM-EKM_OD_pair":"coral", "GRU-EKM_OD_pair":"gold"}

    style = {"LSTM":'o', "GRU":'x', "DBN":"*", "LSTM_OD_pair":"+",
             "GRU_OD_pair":"^", "LSTM-EKM_OD_pair":"D", "GRU-EKM_OD_pair":"<"}

    plt.xlabel("Flow ID, From Largest to Smallest in Mean")
    plt.xlim((0, 529))
    plt.ylabel("Bias(Mbps)")
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)

        plt.scatter(range(data_size), data, marker=style[label], color=color_dict[label], s=20)

    for i in range(len(labels)):
        if labels[i] == "DBN":
            labels[i] = "DBN-OD"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    out_file = "./fig/Bias_OD_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


    # LSTM_EKM_KEC
    labels = ["LSTM", "LSTM_OD_pair", "LSTM-EKM_OD_pair", "LSTM_EKM_KEC_25", "LSTM_EKM_KEC_53", "LSTM_EKM_KEC_106", "LSTM_EKM_KEC_159"]

    color_dict = {"LSTM": "deepskyblue", "LSTM-EKM_OD_pair": "blue", "LSTM_EKM_KEC_25": "orange", "LSTM_EKM_KEC_53": "green",
                  "LSTM_EKM_KEC_106": "purple", "LSTM_EKM_KEC_159":"coral", "LSTM_OD_pair":"red"}

    style = {"LSTM":'o', "LSTM-EKM_OD_pair":'x', "LSTM_EKM_KEC_25":"*", "LSTM_EKM_KEC_53":"+",
             "LSTM_EKM_KEC_106":"^", "LSTM_EKM_KEC_159":"D", "LSTM_OD_pair":"<"}

    plt.xlabel("Flow ID, From Largest to Smallest in Mean")
    plt.xlim((0, 529))
    plt.ylabel("Bias(Mbps)")
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)

        plt.scatter(range(data_size), data, marker=style[label], color=color_dict[label], s=20)

    for i in range(len(labels)):
        if labels[i] == "LSTM_EKM_KEC_25":
            labels[i] = "LSTM-EKM-KEC-5%"
        if labels[i] == "LSTM_EKM_KEC_53":
            labels[i] = "LSTM-EKM-KEC-10%"
        if labels[i] == "LSTM_EKM_KEC_106":
            labels[i] = "LSTM-EKM-KEC-20%"
        if labels[i] == "LSTM_EKM_KEC_159":
            labels[i] = "LSTM-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/Bias_LSTM-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


    # GRU_EKM_KEC
    labels = ["GRU", "GRU_OD_pair", "GRU-EKM_OD_pair", "GRU_EKM_KEC_25", "GRU_EKM_KEC_53", "GRU_EKM_KEC_106", "GRU_EKM_KEC_159"]

    color_dict = {"GRU": "deepskyblue", "GRU-EKM_OD_pair": "blue", "GRU_EKM_KEC_25": "orange", "GRU_EKM_KEC_53": "green",
                  "GRU_EKM_KEC_106": "purple", "GRU_EKM_KEC_159":"coral", "GRU_OD_pair":"red"}

    style = {"GRU":'o', "GRU-EKM_OD_pair":'x', "GRU_EKM_KEC_25":"*", "GRU_EKM_KEC_53":"+",
             "GRU_EKM_KEC_106":"^", "GRU_EKM_KEC_159":"D", "GRU_OD_pair":"<"}

    plt.xlabel("Flow ID, From Largest to Smallest in Mean")
    plt.xlim((0, 529))
    plt.ylabel("Bias(Mbps)")
    ax = plt.gca()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    for label in labels:
        data = np.array(df[label])
        data = data / 1000
        data_size = len(data)

        plt.scatter(range(data_size), data, marker=style[label], color=color_dict[label], s=20)

    for i in range(len(labels)):
        if labels[i] == "GRU_EKM_KEC_25":
            labels[i] = "GRU-EKM-KEC-5%"
        if labels[i] == "GRU_EKM_KEC_53":
            labels[i] = "GRU-EKM-KEC-10%"
        if labels[i] == "GRU_EKM_KEC_106":
            labels[i] = "GRU-EKM-KEC-20%"
        if labels[i] == "GRU_EKM_KEC_159":
            labels[i] = "GRU-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    out_file = "./fig/Bias_GRU-EKM-KEC_" + topology + ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()



# TE MLU bias 结果
# topology: Abilene/CERNET/GEANT
# senario: MCF/SOTE/OSPF
# x0 - x5: CDF 横坐标
def draw_TE_bias(topology, senario, x0, x1, x2, x3, x4, x5):
    file_name = senario + "_MLU_bias_" + topology + ".csv"
    df = pd.read_csv(file_name)
    print(file_name)


    # 直接预测 TM
    labels = ["LSTM", "GRU", "CNN_LSTM", "TCN"]
    color_dict = {"LSTM": "red", "GRU": "blue", "CNN_LSTM": "orange", "TCN": "green"}
    style = {"LSTM": '-', "GRU": '--', "CNN_LSTM": "-.", "TCN": ":"}
    plt.xlabel("Bias of Maximum Link Utilization")
    plt.xlim((x0, x1))
    plt.ylabel("CDF")
    plt.ylim((0, 1))

    width = 1.5
    for label in labels:
        data = np.array(df[label])
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    if senario == "SDN_split":
        out_file = "./fig/" + "MCF"
    elif senario == "hybrid":
        out_file = "./fig/" + "SOTE"
    else:
        out_file = "./fig/" + "OSPF"

    out_file += "_MLU_bias_TM_"
    out_file += topology
    out_file += ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # 预测每个 OD flow，同时对比 LSTM、GRU
    labels = ["LSTM", "GRU", "DBN", "LSTM_OD_pair", "GRU_OD_pair", "LSTM-EKM_OD_pair", "GRU-EKM_OD_pair"]

    color_dict = {"LSTM": "deepskyblue", "GRU": "blue", "DBN": "green", "LSTM_OD_pair": "orange",
                  "GRU_OD_pair": "coral", "LSTM-EKM_OD_pair": "red", "GRU-EKM_OD_pair": "purple"}

    style = {"LSTM":'-', "GRU":'--', "DBN":"-.", "LSTM_OD_pair":":", "GRU_OD_pair":"-.",
             "LSTM-EKM_OD_pair":"--", "GRU-EKM_OD_pair":"-"}

    plt.xlabel("Bias of Maximum Link Utilization")
    plt.xlim((x2, x3))
    plt.ylabel("CDF")
    plt.ylim((0, 1))

    for label in labels:
        if label == "LSTM_OD_pair" or label == "GRU_OD_pair" or \
                        label == "LSTM-EKM_OD_pair" or label == "GRU-EKM_OD_pair":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "DBN":
            labels[i] = "DBN-OD"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
    plt.legend(labels, loc='lower right')
    if senario == "SDN_split":
        out_file = "./fig/" + "MCF"
    elif senario == "hybrid":
        out_file = "./fig/" + "SOTE"
    else:
        out_file = "./fig/" + "OSPF"

    out_file += "_MLU_bias_OD_"
    out_file += topology
    out_file += ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    # LSTM_EKM_KEC
    labels = ["LSTM", "LSTM_OD_pair", "LSTM-EKM_OD_pair", "LSTM_EKM_KEC_25", "LSTM_EKM_KEC_53", "LSTM_EKM_KEC_106",
              "LSTM_EKM_KEC_159"]

    color_dict = {"LSTM": "deepskyblue", "LSTM-EKM_OD_pair": "blue", "LSTM_EKM_KEC_25": "orange",
                  "LSTM_EKM_KEC_53": "green",
                  "LSTM_EKM_KEC_106": "purple", "LSTM_EKM_KEC_159": "coral", "LSTM_OD_pair": "red"}

    style = {"LSTM": '-', "LSTM-EKM_OD_pair": '--', "LSTM_EKM_KEC_25": ":", "LSTM_EKM_KEC_53": "-",
             "LSTM_EKM_KEC_106": "--", "LSTM_EKM_KEC_159": "-.", "LSTM_OD_pair": "-."}

    plt.xlabel("Bias of Maximum Link Utilization")
    plt.xlim((x4, x5))
    plt.ylabel("CDF")
    plt.ylim((0, 1))

    for label in labels:
        if label == "LSTM_EKM_KEC_53" or label == "LSTM_EKM_KEC_106" or label == "LSTM_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "LSTM_EKM_KEC_25":
            labels[i] = "LSTM-EKM-KEC-5%"
        if labels[i] == "LSTM_EKM_KEC_53":
            labels[i] = "LSTM-EKM-KEC-10%"
        if labels[i] == "LSTM_EKM_KEC_106":
            labels[i] = "LSTM-EKM-KEC-20%"
        if labels[i] == "LSTM_EKM_KEC_159":
            labels[i] = "LSTM-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    if senario == "SDN_split":
        out_file = "./fig/" + "MCF"
    elif senario == "hybrid":
        out_file = "./fig/" + "SOTE"
    else:
        out_file = "./fig/" + "OSPF"

    out_file += "_MLU_bias_LSTM-EKM-KEC_"
    out_file += topology
    out_file += ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


    # GRU_EKM_KEC
    labels = ["GRU", "GRU_OD_pair", "GRU-EKM_OD_pair", "GRU_EKM_KEC_25", "GRU_EKM_KEC_53", "GRU_EKM_KEC_106",
              "GRU_EKM_KEC_159"]

    color_dict = {"GRU": "deepskyblue", "GRU-EKM_OD_pair": "blue", "GRU_EKM_KEC_25": "orange",
                  "GRU_EKM_KEC_53": "green",
                  "GRU_EKM_KEC_106": "purple", "GRU_EKM_KEC_159": "coral", "GRU_OD_pair": "red"}

    style = {"GRU": '-', "GRU-EKM_OD_pair": '--', "GRU_EKM_KEC_25": ":", "GRU_EKM_KEC_53": "-",
             "GRU_EKM_KEC_106": "--", "GRU_EKM_KEC_159": "-.", "GRU_OD_pair": "-."}

    plt.xlabel("Bias of Maximum Link Utilization")
    plt.xlim((x4, x5))
    plt.ylabel("CDF")
    plt.ylim((0, 1))

    for label in labels:
        if label == "GRU_EKM_KEC_53" or label == "GRU_EKM_KEC_106" or label == "GRU_EKM_KEC_159":
            width = 2.5
        else:
            width = 1.5

        data = np.array(df[label])
        data_size = len(data)
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[0:-1], cdf, linestyle=style[label], linewidth=width, color=color_dict[label])

    for i in range(len(labels)):
        if labels[i] == "GRU_EKM_KEC_25":
            labels[i] = "GRU-EKM-KEC-5%"
        if labels[i] == "GRU_EKM_KEC_53":
            labels[i] = "GRU-EKM-KEC-10%"
        if labels[i] == "GRU_EKM_KEC_106":
            labels[i] = "GRU-EKM-KEC-20%"
        if labels[i] == "GRU_EKM_KEC_159":
            labels[i] = "GRU-EKM-KEC-30%"
        if labels[i] == "LSTM":
            labels[i] = "LSTM-TM"
        if labels[i] == "GRU":
            labels[i] = "GRU-TM"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"

    plt.legend(labels, loc='lower right')
    if senario == "SDN_split":
        out_file = "./fig/" + "MCF"
    elif senario == "hybrid":
        out_file = "./fig/" + "SOTE"
    else:
        out_file = "./fig/" + "OSPF"

    out_file += "_MLU_bias_GRU-EKM-KEC_"
    out_file += topology
    out_file += ".png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    out_path = "./fig/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # draw_TM_RMSE("RMSE_result.csv", "GEANT")
    # draw_TM_MAE("MAE_result.csv", "GEANT")
    # draw_OD_bias("bias_OD_result.csv", "GEANT")

    # draw_TE_bias("GEANT", "SDN_split", 0, 2, 0, 2, 0, 2)
    draw_TE_bias("GEANT", "OSPF", -0.3, 1, -0.3, 1, -0.3, 1)
    # draw_TE_bias("GEANT", "hybrid", -0.2, 0.7, -0.2, 0.7, -0.2, 0.7)
import re
import csv
import os
import time
import numpy as np
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

CAPA = 26000
PATH_CERNET = "../TM_result/CERNET/"
Origin_TM = PATH_CERNET + "Origin/"

def get_MAE_RMSE(list1, list2, length):
    MAE = 0
    RMSE = 0
    for i in range(length):
        MAE += abs(list1[i] - list2[i])
        RMSE += pow((list1[i] - list2[i]), 2)
    MAE /= length
    RMSE /= length
    RMSE = math.sqrt(RMSE)
    return MAE, RMSE

def cdf(data):
    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    label = ["LSTM_KEY_CORRECT"]
    plt.plot(bin_edges[0:-1], cdf, linestyle='--', color='orangered')
    plt.legend(label, loc='upper left')
    plt.xlim((0, 2))
    plt.ylim((0,1))
    plt.ylabel("CDF")
    # plt.grid(True)

    plt.show()

def draw_cdf(file_name, out_file):
    color_dict = {"Origin": "red", "LSTM": "deepskyblue", "CNN_LSTM": "orange", "LSTM_OD_pair": "seagreen",
                  "TCN": "slategray",
                  "LSTM_KEY_CORRECT": "purple", "DBN": "brown", "GRU": "blue", "GRU_OD_pair": "tomato",
                  "GRU_KEY_CORRECT": "black"}
    df = pd.read_csv(file_name)
    count = 0
    plt.xlim((7, 100))
    plt.xlabel("Maximum link utilization")
    plt.ylim((0,1))
    plt.ylabel("CDF")


    labels = ["Origin", "LSTM", "GRU", "DBN", "TCN", "CNN_LSTM", "LSTM_OD_pair", "LSTM_KEY_CORRECT",
              "GRU_OD_pair", "GRU_KEY_CORRECT"]

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
        plt.plot(bin_edges[0:-1], cdf, linestyle='-', color=color_dict[label])

        count += 1


    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if "_KEY_CORRECT" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-KEC"
        if labels[i] == "Origin":
            labels[i] = "Actual"

    plt.legend(labels, loc='lower right')
    plt.savefig(out_file)
    plt.show()

def draw_cdf_bias(file_name, out_file):
    color_dict = {"Origin": "red", "LSTM": "deepskyblue", "CNN_LSTM": "orange", "LSTM_OD_pair": "seagreen",
                  "TCN": "slategray",
                  "LSTM_KEY_CORRECT": "purple", "DBN": "brown", "GRU": "blue", "GRU_OD_pair": "tomato",
                  "GRU_KEY_CORRECT": "red"}
    df = pd.read_csv(file_name)
    plt.xlim((-0.5, 1))
    plt.xlabel("bias of maximum link utilization")
    plt.ylim((0,1))
    plt.ylabel("CDF")
    count = 0

    # labels = ["GRU", "LSTM", "TCN", "CNN_LSTM", "DBN"]
    labels = ["GRU", "LSTM", "GRU_OD_pair", "LSTM_OD_pair", "GRU_KEY_CORRECT", "LSTM_KEY_CORRECT"]
    style = {"GRU":'-', "LSTM":'--', "TCN":"-.", "CNN_LSTM":":", "DBN":"-", "GRU_OD_pair":'-.', "LSTM_OD_pair":':',
                 "GRU_KEY_CORRECT":"-","LSTM_KEY_CORRECT":'--'}

    for label in labels:
        if label == "DBN" or label == "GRU_KEY_CORRECT" or label == "LSTM_KEY_CORRECT":
            width = 3.8
        else:
            width = 2

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

        count += 1


    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if "_KEY_CORRECT" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-KEC"

    plt.legend(labels, loc='lower right')
    # plt.savefig("hybrid_MLU_bias_diff_model.png", bbox_inches='tight')
    plt.savefig("hybrid_MLU_bias_diff_pers.png", bbox_inches='tight')
    plt.show()

def draw_MAE_RMSE(file_name, out_file):
    color_dict = {"Origin":"red", "LSTM":"deepskyblue", "LRCN":"orange", "LSTM-OD":"seagreen", "TCN":"slategray",
                  "LSTM-KEC":"purple", "DBN":"brown", "GRU":"blue", "GRU-OD":"tomato", "GRU-KEC":"black"}

    labels = ["GRU-OD", "LSTM-OD", "GRU-KEC", "LSTM-KEC", "GRU", "LSTM", "DBN",
              "TCN", "LRCN"]

    index_dict = {"GRU-OD":9, "LSTM-OD":10, "GRU-KEC":4, "LSTM-KEC":6, "GRU":8, "LSTM":2,
                  "DBN":7, "TCN":5, "LRCN":3}

    result = []  # MAE, RMSE
    for i in range(11):
        result.append([])
    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        for row in csv_reader:
            if row[0] == "Data":
                continue
            for label in labels:
                result[index_dict[label]].append(float(row[index_dict[label]]))

    for label in labels:
        print(result[index_dict[label]])

    fig, ax = plt.subplots()
    x = [1, 2]
    bar_width = 0.095
    x = np.arange(len(x))
    opacity = 0.9

    count = 0
    for label in labels:
        rects = ax.bar(x + count * bar_width, result[index_dict[label]], bar_width,
                        alpha=opacity, color=color_dict[label], label=label)
        count += 1



    ax.set_xticks(x + 9 * bar_width / 2)
    ax.set_xticklabels(("MAE", "RMSE"))
    plt.xlabel("MAE and RMSE between MLU and MLU'")
    plt.ylabel("value")
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(out_file)
    plt.show()

def write_row_to_csv(data, file_name):
    with open(file_name, 'a+', newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(data)



def get_TM_volume(TM_file):
    result = 0
    with open(TM_file, 'r') as reader:
        while True:
            line = reader.readline().strip()
            if not line:
                break

            if not line == ' ':
                line = line.split(' ')
                src = int(line[0]) - 1
                dst = int(line[1]) - 1
                value = float(line[2])
                result += value
    return result / 1000

# 分析什么情况下，MLU超出阈值
def analysis_overflow(file_name, threshold, nums):
    # 关注LSTM_OD、GRU_OD
    overflow_TM = [[], []]
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "Data":
                continue
            if float(row[3]) > threshold:  # LSTM-OD
                overflow_TM[0].append(int(row[0]))
            if float(row[6]) > threshold:  # GRU-OD
                overflow_TM[1].append(int(row[0]))

    print(overflow_TM)

    # 分析TM总流量
    origin_TM = []
    LSTM_OD_TM = []
    GRU_OD_TM = []
    for i in range(nums):
        origin_TM_path = Origin_TM + "Origin_" + str(i + 1) + ".txt"
        LSTM_OD_TM_path = PATH_CERNET + "LSTM_OD_pair/LSTM_OD_pair_" + str(i + 1) + ".txt"
        GRU_OD_TM_path = PATH_CERNET + "GRU_OD_pair/GRU_OD_pair_" + str(i + 1) + ".txt"

        origin_TM.append(get_TM_volume(origin_TM_path))
        LSTM_OD_TM.append(get_TM_volume(LSTM_OD_TM_path))
        GRU_OD_TM.append(get_TM_volume(GRU_OD_TM_path))


    x_axix = []
    for i in range(nums):
        x_axix.append(i + 1)

    plt.title('Overflow Analysis')
    plt.plot(x_axix, origin_TM, color='green', label='Actual volume')
    # plt.plot(x_axix, LSTM_OD_TM,  color='red', label='LSTM-OD volume')
    plt.plot(x_axix, GRU_OD_TM, color='blue', label='GRU-OD volume')
    plt.legend()

    # for index in overflow_TM[0]:
    #     plt.axvline(int(index), linestyle='--', color='red', alpha=0.5)
    for index in overflow_TM[1]:
        plt.axvline(int(index), linestyle='--', color='red', alpha=0.5)
        # plt.axvline(index)

    # plt.axvline(210, linestyle='--', color='blue')
    # plt.axvline(240, linestyle='--', color='blue')


    plt.xlabel('TM number')
    plt.ylabel('Volume(Mbps)')
    plt.savefig('overflow_volume_MCF.png', bbox_inches='tight')
    plt.show()
    plt.close()

    '''
    # 分析总流量差值
    bias_LSTM_OD = []
    bias_GRU_OD = []
    for i in range(nums):
        bias_LSTM_OD.append(abs(origin_TM[i] - LSTM_OD_TM[i]))
        bias_GRU_OD.append(abs(origin_TM[i] - GRU_OD_TM[i]))

    plt.title('Overflow Analysis')
    # plt.plot(x_axix, bias_LSTM_OD,  color='red', label='LSTM-OD bias')
    plt.plot(x_axix, bias_GRU_OD, color='blue', label='GRU-OD bias')
    plt.legend()

    # for index in overflow_TM[0]:
    #     plt.axvline(int(index), linestyle='--', color='red', alpha=0.5)
    for index in overflow_TM[1]:
        plt.axvline(int(index), linestyle='--', color='red', alpha=0.5)
        # plt.axvline(index)

    # plt.axvline(210, linestyle='--', color='blue')
    # plt.axvline(240, linestyle='--', color='blue')


    plt.xlabel('TM number')
    plt.ylabel('bias')
    plt.savefig('overflow_bias_MCF.png', bbox_inches='tight')
    plt.show()
    plt.close()
    

    # 观察训练集的TM volume曲线，看看如何选择volume阈值
    file_name = "../OD_pair/CERNET-OD_pair_2013-03-01.csv"
    volume_list = []
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "time":
                continue
            if row[0] == '':
                break
            volume = 0
            for i in range(1, len(row)):
                volume += float(row[i])
            volume_list.append(volume)
    length = len(volume_list)


    x_axix = []
    for i in range(length):
        x_axix.append(i + 1)
    plt.title('TM Volume Analysis')
    # plt.plot(x_axix, bias_LSTM_OD,  color='red', label='LSTM-OD bias')
    plt.plot(x_axix, volume_list, color='blue', label='TM volume')
    plt.legend()
    plt.axvline(length - nums, linestyle='-', color='purple')  # 训练集-测试集分界线
    for index in overflow_TM[1]:
        plt.axvline(length - nums + int(index), linestyle='--', color='red', alpha=0.5)
        # plt.axvline(index)

    # 对训练集volume进行排序，然后选择阈值S，保证S大于w%的volume，这里 w% 这个取值需要根据训练集数据的分布情况来决定
    # 训练集数据较稳定，则w取值较高；训练集就有很多burst，则取值相对调低
    # 如何定义一个算法？
    volume_list = sorted(volume_list)
    s = volume_list[int(length * 0.99)]
    plt.axhline(s, linestyle='-', color='green', alpha=0.5)


    plt.xlabel('TM number')
    plt.ylabel('Volume')
    plt.savefig('overflow_total_TM_volume_MCF.png', bbox_inches='tight')
    plt.show()
    '''

def generate_bias_and_MAE(file_name):
    result_list = []
    first_line = []
    for i in range(22):
        result_list.append([])
    nums = 1238
    with open(file_name) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            if row[0] == "Data":
                first_line = row
                continue
            for i in range(1, len(row)):
                result_list[i - 1].append(float(row[i]))
    MAE_result = ["MAE", "-"]
    RMSE_result = ["RMSE", "-"]

    for i in range(1, 22):
        MAE, RMSE = get_MAE_RMSE(result_list[0], result_list[i], nums)
        MAE_result.append(MAE)
        RMSE_result.append(RMSE)
    out_file = "hybrid_MLU_MAE_RMSE.csv"
    write_row_to_csv(MAE_result, out_file)
    write_row_to_csv(RMSE_result, out_file)

    out_file = "hybrid_MLU_bias_CERNET.csv"
    write_row_to_csv(first_line, out_file)

    for i in range(nums):
        data = [i + 1, 0]
        for j in range(1, len(result_list)):
            data.append((result_list[j][i] - result_list[0][i]) / result_list[0][i])
        write_row_to_csv(data, out_file)


if __name__ == "__main__":
    generate_bias_and_MAE("hybrid_MLU_CERNET.csv")

    # draw cdf
    # draw_cdf("hybrid_MLU_CERNET.csv", "hybrid_MLU_result.png")
    # draw_cdf_bias("hybrid_MLU_bias_CERNET.csv", "hybrid_MLU_bias.png")
    # draw_MAE_RMSE("hybrid_MLU_MAE_RMSE.csv", "hybrid_MLU_MAE_RMSE.png")


    # analysis_overflow("overflow_hybrid_CERNET.CSV", 85, 1238)
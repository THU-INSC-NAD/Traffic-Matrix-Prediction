import re
import csv
import os
import time
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

CAPA = 9920000
PATH_CERNET = "../TM_result/CERNET/"

# TODO where to find there files?
PATH_OSPF_CERNET = "C:/Microsoft Visual Studio 11.0/workspace/OSPF/SOTE/CERNET/"
PATH_HYBRID_CERNET = "E:/java_workspace/SOTE/TM_predict/CERNET/"

Origin_TM = PATH_CERNET + "Origin/"
LSTM_TM = PATH_CERNET + "LSTM/"
CNN_LSTM_TM = PATH_CERNET + "CNN_LSTM/"
TCN_TM = PATH_CERNET + "TCN/"
LSTM_KEY_CORRECT_TM = PATH_CERNET + "LSTM_KEY_CORRECT/"
LSTM_OD_pair_TM = PATH_CERNET + "LSTM_OD_pair/"
DBN_TM = PATH_CERNET + "DBN/"
GRU_TM = PATH_CERNET + "GRU/"
GRU_OD_pair_TM = PATH_CERNET + "GRU_OD_pair/"
GRU_KEY_CORRECT_TM = PATH_CERNET + "GRU_KEY_CORRECT/"

LSTM_EKM_OD_TM = PATH_CERNET + "LSTM-EKM_OD_pair/"
LSTM_EKM_KEC_10_TM = PATH_CERNET + "LSTM_EKM_KEC_10/"
LSTM_EKM_KEC_20_TM = PATH_CERNET + "LSTM_EKM_KEC_20/"
LSTM_EKM_KEC_40_TM = PATH_CERNET + "LSTM_EKM_KEC_40/"
LSTM_EKM_KEC_78_TM = PATH_CERNET + "LSTM_EKM_KEC_78/"
LSTM_KEC_10_TM = PATH_CERNET + "LSTM_KEC_10/"
LSTM_KEC_20_TM = PATH_CERNET + "LSTM_KEC_20/"
LSTM_KEC_40_TM = PATH_CERNET + "LSTM_KEC_40/"
LSTM_KEC_78_TM = PATH_CERNET + "LSTM_KEC_78/"
GRU_EKM_OD_TM = PATH_CERNET + "GRU-EKM_OD_pair/"
GRU_EKM_KEC_10_TM = PATH_CERNET + "GRU_EKM_KEC_10/"
GRU_EKM_KEC_20_TM = PATH_CERNET + "GRU_EKM_KEC_20/"
GRU_EKM_KEC_40_TM = PATH_CERNET + "GRU_EKM_KEC_40/"
GRU_EKM_KEC_78_TM = PATH_CERNET + "GRU_EKM_KEC_78/"

Origin_OSPF_TM = PATH_OSPF_CERNET + "Origin/"
LSTM_OSPF_TM = PATH_OSPF_CERNET + "LSTM/"
CNN_LSTM_OSPF_TM = PATH_OSPF_CERNET + "CNN_LSTM/"
TCN_OSPF_TM = PATH_OSPF_CERNET + "TCN/"
LSTM_KEY_CORRECT_OSPF_TM = PATH_OSPF_CERNET + "LSTM_KEY_CORRECT/"
LSTM_OD_pair_OSPF_TM = PATH_OSPF_CERNET + "LSTM_OD_pair/"
DBN_OSPF_TM = PATH_OSPF_CERNET + "DBN/"
GRU_OSPF_TM = PATH_OSPF_CERNET + "GRU/"
GRU_OD_pair_OSPF_TM = PATH_OSPF_CERNET + "GRU_OD_pair/"
GRU_KEY_CORRECT_OSPF_TM = PATH_OSPF_CERNET + "GRU_KEY_CORRECT/"

LSTM_EKM_OD_OSPF_TM = PATH_OSPF_CERNET + "LSTM-EKM_OD_pair/"
LSTM_EKM_KEC_10_OSPF_TM = PATH_OSPF_CERNET + "LSTM_EKM_KEC_10/"
LSTM_EKM_KEC_20_OSPF_TM = PATH_OSPF_CERNET + "LSTM_EKM_KEC_20/"
LSTM_EKM_KEC_40_OSPF_TM = PATH_OSPF_CERNET + "LSTM_EKM_KEC_40/"
LSTM_EKM_KEC_78_OSPF_TM = PATH_OSPF_CERNET + "LSTM_EKM_KEC_78/"
LSTM_KEC_10_OSPF_TM = PATH_OSPF_CERNET + "LSTM_KEC_10/"
LSTM_KEC_20_OSPF_TM = PATH_OSPF_CERNET + "LSTM_KEC_20/"
LSTM_KEC_40_OSPF_TM = PATH_OSPF_CERNET + "LSTM_KEC_40/"
LSTM_KEC_78_OSPF_TM = PATH_OSPF_CERNET + "LSTM_KEC_78/"
GRU_EKM_OD_OSPF_TM = PATH_OSPF_CERNET + "GRU-EKM_OD_pair/"
GRU_EKM_KEC_10_OSPF_TM = PATH_OSPF_CERNET + "GRU_EKM_KEC_10/"
GRU_EKM_KEC_20_OSPF_TM = PATH_OSPF_CERNET + "GRU_EKM_KEC_20/"
GRU_EKM_KEC_40_OSPF_TM = PATH_OSPF_CERNET + "GRU_EKM_KEC_40/"
GRU_EKM_KEC_78_OSPF_TM = PATH_OSPF_CERNET + "GRU_EKM_KEC_78/"

Origin_HYBRID_TM = PATH_HYBRID_CERNET + "Origin/"
LSTM_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM/"
CNN_LSTM_HYBRID_TM = PATH_HYBRID_CERNET + "CNN_LSTM/"
TCN_HYBRID_TM = PATH_HYBRID_CERNET + "TCN/"
LSTM_KEY_CORRECT_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_KEY_CORRECT/"
LSTM_OD_pair_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_OD_pair/"
DBN_HYBRID_TM = PATH_HYBRID_CERNET + "DBN/"
GRU_HYBRID_TM = PATH_HYBRID_CERNET + "GRU/"
GRU_OD_pair_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_OD_pair/"
GRU_KEY_CORRECT_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_KEY_CORRECT/"

LSTM_EKM_OD_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM-EKM_OD_pair/"
LSTM_EKM_KEC_10_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_EKM_KEC_10/"
LSTM_EKM_KEC_20_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_EKM_KEC_20/"
LSTM_EKM_KEC_40_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_EKM_KEC_40/"
LSTM_EKM_KEC_78_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_EKM_KEC_78/"
LSTM_KEC_10_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_KEC_10/"
LSTM_KEC_20_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_KEC_20/"
LSTM_KEC_40_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_KEC_40/"
LSTM_KEC_78_HYBRID_TM = PATH_HYBRID_CERNET + "LSTM_KEC_78/"
GRU_EKM_OD_HYBRID_TM = PATH_HYBRID_CERNET + "GRU-EKM_OD_pair/"
GRU_EKM_KEC_10_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_EKM_KEC_10/"
GRU_EKM_KEC_20_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_EKM_KEC_20/"
GRU_EKM_KEC_40_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_EKM_KEC_40/"
GRU_EKM_KEC_78_HYBRID_TM = PATH_HYBRID_CERNET + "GRU_EKM_KEC_78/"


def generate_topo(topo_file, out_file):
    f = open(out_file, 'w')
    with open(topo_file, 'r') as readTxt:
        while True:
            line = readTxt.readline()
            if not line:
                break
            line = line.strip()
            if line.count(' ') == 1:  # node number and edge number
                line = line.split()
                node_num = int(line[0])
                edge_num = int(line[1])
                f.write("NUM-NODES: " + str(node_num) + '\n')
            else:  # src dst capa weight
                line = line.split()
                src = int(line[0])
                dst = int(line[1])
                capa = int(line[2])
                weight = int(line[3])
                f.write("LINK: " + str(src) + ' ' + str(dst) + ' ' + "CC 1\n")

def generate_tm(TM_file, out_file, node):
    TM = np.zeros(shape=(node, node))
    with open(TM_file, 'r') as readTxt:
        while True:
            line = readTxt.readline()
            if not line:
                break
            # line : OD-src OD-dst link-src link-dst link-value
            line = line.split()
            OD_src = int(line[0]) - 1
            OD_dst = int(line[1]) - 1
            traffic = float(line[2])
            TM[OD_src][OD_dst] = traffic

    f = open(out_file, 'w')
    for i in range(node):
        data = ' '
        for j in range(node - 1):
            data += str(TM[i][j])
            data += ' '
        data += str(TM[i][node - 1])
        if not i == node - 1:
            data += '\n'
        f.write(data)



def generate_TMs(in_path, out_path, nodes):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    files = os.listdir(in_path)
    for file in files:
        print(file)
        in_file_name = in_path + file
        out_file_name = out_path + file
        generate_tm(in_file_name, out_file_name, nodes)


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

def generate_bias_and_MAE(file_name):
    result_list = []
    first_line = []
    for i in range(22):
        result_list.append([])
    nums = 1238
    with open("OSPF_MLU_CERNET.csv") as csvfile:
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
    out_file = "OSPF_MLU_MAE_RMSE.csv"
    write_row_to_csv(MAE_result, out_file)
    write_row_to_csv(RMSE_result, out_file)

    out_file = "OSPF_MLU_bias_CERNET.csv"
    write_row_to_csv(first_line, out_file)

    for i in range(nums):
        data = [i + 1, 0]
        for j in range(1, len(result_list)):
            data.append((result_list[j][i] - result_list[0][i]) / result_list[0][i])
        write_row_to_csv(data, out_file)

def draw_cdf(file_name, out_file):
    color_dict = {"Origin": "red", "LSTM": "deepskyblue", "CNN_LSTM": "orange", "LSTM_OD_pair": "seagreen",
                  "TCN": "slategray",
                  "LSTM_KEY_CORRECT": "purple", "DBN": "brown", "GRU": "blue", "GRU_OD_pair": "tomato",
                  "GRU_KEY_CORRECT": "black"}
    df = pd.read_csv(file_name)
    count = 0
    plt.xlim((5, 100))
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
    plt.xlim((-0.4, 1))
    plt.xlabel("bias of maximum link utilization")
    plt.ylim((0,1))
    plt.ylabel("CDF")
    count = 0

    labels = ["LSTM", "GRU", "DBN", "TCN", "CNN_LSTM", "LSTM_OD_pair", "LSTM_KEY_CORRECT",
              "GRU_OD_pair", "GRU_KEY_CORRECT"]

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
    # plt.savefig("OSPF_MLU_diff_models.png", bbox_inches='tight')
    plt.savefig("OSPF_MLU_diff_persp.png", bbox_inches='tight')
    plt.show()


def draw_MAE_RMSE(file_name, out_file):
    color_dict = {"Origin":"red", "LSTM":"deepskyblue", "LRCN":"orange", "LSTM-OD":"seagreen", "TCN":"slategray",
                  "LSTM-KEC":"purple", "DBN":"brown", "GRU":"blue", "GRU-OD":"tomato", "GRU-KEC":"black"}

    labels = ["GRU-OD", "LSTM-OD", "GRU-KEC", "LSTM-KEC", "GRU", "LSTM", "DBN",
              "TCN", "LRCN"]

    index_dict = {"GRU-OD":9, "LSTM-OD":10, "GRU-KEC":4, "LSTM-KEC":6, "GRU":8, "LSTM":2,
                  "DBN":7, "TCN":5, "LRCN":3}

    result = [] # MAE, RMSE
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
    plt.savefig('overflow_volume_OSPF.png', bbox_inches='tight')
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
    plt.savefig('overflow_bias_OSPF.png', bbox_inches='tight')
    plt.show()
    plt.close()
    '''

    '''
    # 观察训练集的TM volume曲线，看看如何选择volume阈值
    file_name = "E:/Tsinghua/master/Project/code/traffic matrix prediction/OD_pair/CERNET-OD_pair_2013-03-01.csv"
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
    # 相当于这里要建立分布模型？
    volume_list = sorted(volume_list)
    s = volume_list[int(length * 0.95)]
    plt.axhline(s, linestyle='-', color='green', alpha=0.5)


    plt.xlabel('TM number')
    plt.ylabel('Volume')
    plt.savefig('overflow_total_TM_volume_OSPF.png', bbox_inches='tight')
    plt.show()
    '''


if __name__ == "__main__":
    # generate TMs for c++ codes
    # generate_topo("CERNET.txt", "ospf_CERNET.txt")
    # generate_topo("cernet.txt", "cernet.topo")
    # in_path_list = [Origin_TM, LSTM_TM, CNN_LSTM_TM, TCN_TM, LSTM_KEY_CORRECT_TM, LSTM_OD_pair_TM, DBN_TM, GRU_TM, GRU_OD_pair_TM, GRU_KEY_CORRECT_TM]
    # out_path_list = [Origin_OSPF_TM, LSTM_OSPF_TM, CNN_LSTM_OSPF_TM, TCN_OSPF_TM, LSTM_KEY_CORRECT_OSPF_TM, LSTM_OD_pair_OSPF_TM,
    #                  DBN_OSPF_TM, GRU_OSPF_TM, GRU_OD_pair_OSPF_TM, GRU_KEY_CORRECT_OSPF_TM]

    # in_path_list = [GRU_EKM_OD_TM, GRU_EKM_KEC_10_TM, GRU_EKM_KEC_20_TM, GRU_EKM_KEC_40_TM, GRU_EKM_KEC_78_TM]
    #
    # # out_path_list = [GRU_EKM_OD_OSPF_TM, GRU_EKM_KEC_10_OSPF_TM, GRU_EKM_KEC_20_OSPF_TM,
    # #                  GRU_EKM_KEC_40_OSPF_TM, GRU_EKM_KEC_78_OSPF_TM]
    #
    # out_path_list = [GRU_EKM_OD_HYBRID_TM, GRU_EKM_KEC_10_HYBRID_TM, GRU_EKM_KEC_20_HYBRID_TM,
    #                  GRU_EKM_KEC_40_HYBRID_TM, GRU_EKM_KEC_78_HYBRID_TM]
    #
    # in_path_list = [LSTM_OD_pair_TM]
    # # out_path_list = [LSTM_OD_pair_OSPF_TM]
    # out_path_list = [LSTM_OD_pair_HYBRID_TM]
    # for i in range(len(in_path_list)):
    #     generate_TMs(in_path_list[i], out_path_list[i], 14)


    # get bias and MAE of MLU' and MLU
    generate_bias_and_MAE("OSPF_MLU_CERNET.csv")

    # draw results
    # draw_cdf("OSPF_MLU_CERNET.csv", "OSPF_MLU_result.png")
    # draw_cdf_bias("OSPF_MLU_bias_CERNET.csv", "OSPF_MLU_bias.png")
    # draw_MAE_RMSE("OSPF_MLU_MAE_RMSE.csv", "OSPF_MLU_MAE_RMSE.png")

    # overflow analysis
    # analysis_overflow("overflow_OSPF_CERNET.csv", 105, 1238)
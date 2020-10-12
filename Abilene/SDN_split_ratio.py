import re
import csv
import os
import time
import numpy as np
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

CAPA = 18500
PATH_Abilene = "../TM_result/Abilene/"
STRATEGY_ORIGIN = PATH_Abilene + "split_ratio/Origin/"
STRATEGY_LSTM = PATH_Abilene + "split_ratio/LSTM/"
STRATEGY_CNN_LSTM = PATH_Abilene + "split_ratio/CNN_LSTM/"
STRATEGY_LSTM_OD = PATH_Abilene + "split_ratio/LSTM_OD_pair/"
STRATEGY_TCN = PATH_Abilene + "split_ratio/TCN/"
STRATEGY_LSTM_KEY_correct = PATH_Abilene + "split_ratio/LSTM_KEY_CORRECT/"
STRATEGY_DBN = PATH_Abilene + "split_ratio/DBN/"
STRATEGY_GRU = PATH_Abilene + "split_ratio/GRU/"
STRATEGY_GRU_OD = PATH_Abilene + "split_ratio/GRU_OD_pair/"
STRATEGY_GRU_KEY_CORRECT = PATH_Abilene + "split_ratio/GRU_KEY_CORRECT/"

STRATEGY_LSTM_EKM_OD = PATH_Abilene + "split_ratio/LSTM-EKM_OD_pair/"
STRATEGY_LSTM_EKM_KEC_7 = PATH_Abilene + "split_ratio/LSTM_EKM_KEC_7/"
STRATEGY_LSTM_EKM_KEC_14 = PATH_Abilene + "split_ratio/LSTM_EKM_KEC_14/"
STRATEGY_LSTM_EKM_KEC_28 = PATH_Abilene + "split_ratio/LSTM_EKM_KEC_28/"
STRATEGY_LSTM_EKM_KEC_43 = PATH_Abilene + "split_ratio/LSTM_EKM_KEC_43/"
STRATEGY_LSTM_KEC_7 = PATH_Abilene + "split_ratio/LSTM_KEC_7/"
STRATEGY_LSTM_KEC_14 = PATH_Abilene + "split_ratio/LSTM_KEC_14/"
STRATEGY_LSTM_KEC_28 = PATH_Abilene + "split_ratio/LSTM_KEC_28/"
STRATEGY_LSTM_KEC_43 = PATH_Abilene + "split_ratio/LSTM_KEC_43/"

STRATEGY_GRU_EKM_OD = PATH_Abilene + "split_ratio/GRU-EKM_OD_pair/"
STRATEGY_GRU_EKM_KEC_7 = PATH_Abilene + "split_ratio/GRU_EKM_KEC_7/"
STRATEGY_GRU_EKM_KEC_14 = PATH_Abilene + "split_ratio/GRU_EKM_KEC_14/"
STRATEGY_GRU_EKM_KEC_28 = PATH_Abilene + "split_ratio/GRU_EKM_KEC_28/"
STRATEGY_GRU_EKM_KEC_43 = PATH_Abilene + "split_ratio/GRU_EKM_KEC_43/"

Origin_TM = PATH_Abilene + "Origin/"

def construct_graph(topo_file):
    G = nx.DiGraph()
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
                # print(node_num, edge_num)
                for i in range(node_num):
                    G.add_node(i)
            else:  # src dst capa weight
                line = line.split()
                src = int(line[0])
                dst = int(line[1])
                capa = int(line[2])
                weight = int(line[3])
                G.add_edge(src, dst)
                G[src][dst]['traffic'] = 0.0
                G.add_edge(dst, src)
                G[dst][src]['traffic'] = 0.0
                # print(src, dst, capa, weight)
    return G

def get_routing_strategy(TM_file, flow_file):
    # get origin TM demand
    tm_dict = {}  # O-D: traffic
    with open(TM_file, 'r') as readTxt:
        while True:
            line = readTxt.readline()
            if not line:
                break
            # line : OD-src OD-dst link-src link-dst link-value
            line = line.split()
            OD_src = int(line[0]) - 1
            OD_dst = int(line[1]) - 1
            if OD_src == OD_dst:
                continue
            traffic = float(line[2])
            temp = str(OD_src) + '-' + str(OD_dst)

            tm_dict[temp] = traffic

    # calculate strategy based on TM and split flow_file
    link_dict = {}  # OD : {link: ratio}
    with open(flow_file, 'r') as readTxt:
        while True:
            line = readTxt.readline()
            if not line:
                break
            # line : OD-src OD-dst link-src link-dst link-value
            line = line.split()
            OD_src = int(line[0])
            OD_dst = int(line[1])
            link_src = int(line[2])
            link_dst = int(line[3])
            link_value = float(line[4])
            # print(OD_src, OD_dst, link_src, link_dst, link_value)
            temp = str(OD_src) + '-' + str(OD_dst)

            if not temp in tm_dict:
                continue

            key = str(OD_src) + '-' + str(OD_dst)

            if not key in link_dict:
                link_dict[key] = {}
            link = str(link_src) + '-' + str(link_dst)
            if not link in link_dict[key]:
                link_dict[key][link] = 0
            link_dict[key][link] += link_value
    # print(link_dict)

    # remove cycle from cplex
    for key in link_dict:
        G = nx.DiGraph()
        node_list = []
        for link in link_dict[key]:
            link_src = int(link.split('-')[0])
            link_dst = int(link.split('-')[1])

            if not link_src in node_list:
                node_list.append(link_src)
                G.add_node(link_src)
            if not link_dst in node_list:
                node_list.append(link_dst)
                G.add_node(link_dst)
            G.add_edge(link_src, link_dst)

        cycle_list = list(nx.simple_cycles(G))
        if cycle_list:
            OD_src = int(key.split('-')[0])
            OD_dst = int(key.split('-')[1])
            # print(OD_src, OD_dst, cycle_list)

            full_path_list = nx.all_simple_paths(G, source=OD_src, target=OD_dst)
            path_list = []
            for path in full_path_list:
                length = len(path)
                temp = ''
                for i in range(length - 1):
                    temp += str(path[i])
                    temp += '-'
                temp += str(path[length - 1])
                path_list.append(temp)
            # print(OD_src, OD_dst, path_list)


            for cycle in cycle_list:
                min_value = 9999999999999
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    link = str(src) + '-' + str(dst)
                    if min_value > link_dict[key][link]:
                        min_value = link_dict[key][link]

                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    link = str(src) + '-' + str(dst)
                    link_dict[key][link] -= min_value
                    for path in path_list:
                        if not link in path:
                            link_dict[key][link] = 0


            # for i in cycle_list:
            #     src = i[0]
            #     dst = i[1]
            #     link = str(src) + '-' + str(dst)
            #     if min_value > link_dict[key][link]:
            #         min_value = link_dict[key][link]
            # print(OD_src, OD_dst, min_value)
            # for i in cycle_list:
            #     src = i[0]
            #     dst = i[1]
            #     link = str(src) + '-' + str(dst)
            #     link_dict[key][link] -= min_value
            #     for path in path_list:
            #         if not link in path:
            #             link_dict[key][link] = 0

    # get ratio strategy
    for key in link_dict:
        for link in link_dict[key]:
            link_dict[key][link] /= tm_dict[key]
            # for cplex error
            # if link_dict[key][link] > 1.001:
            #     print(key, link, link_dict[key][link])

            link_dict[key][link] = float(format(link_dict[key][link],'.5f'))

    # print(link_dict["13-0"])
    return link_dict


# route based on strategy
def routing(G, TM_file, strategy, out_file):
    # read TM
    # OD-src OD-dst traffic
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

            key = str(OD_src) + '-' + str(OD_dst)
            if key in strategy:
                for link in strategy[key]:
                    # print(link)
                    link_src = int(link.split('-')[0])
                    link_dst = int(link.split('-')[1])
                    volume = traffic * float(strategy[key][link])
                    G[link_src][link_dst]["traffic"] += volume

                '''
                # route based on strategy
                full_path_list = nx.all_simple_paths(G, source=OD_src, target=OD_dst)
                for path in full_path_list:
                    if str(path) in strategy[key]:
                        for i in range(len(path) - 1):
                            G[path[i]][path[i + 1]]["traffic"] += traffic * strategy[key][str(path)]
                '''
            else:
                # continue
                # route based on shortest path
                shortest_path = nx.shortest_path(G, source=OD_src, target=OD_dst)
                for i in range(len(shortest_path) - 1):
                    G[shortest_path[i]][shortest_path[i + 1]]["traffic"] += traffic


    sorted_edge = sorted(G.edges(data=True), key=lambda x: x[2]["traffic"], reverse=True)
    # MLU = sorted_edge[0][2]["traffic"] / CAPA * 100
    MLU = sorted_edge[0][2]["traffic"] / CAPA
    # print("MLU:", MLU)
    return MLU


def write_row_to_csv(data, file_name):
    with open(file_name, 'a+', newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(data)


def get_OD_list(Origin_TM, TM):
    OD_list = []
    with open(Origin_TM, 'r') as reader:
        while True:
            line = reader.readline().strip()
            if not line:
                break
            if not line == ' ':
                line = line.split(' ')
                src = int(line[0])
                dst = int(line[1])
                value = float(line[2])
                temp = str(src) + '-' + str(dst)
                OD_list.append(temp)

    f = open("test.txt", 'w')
    with open(TM, 'r') as reader:
        while True:
            line = reader.readline().strip()
            if not line:
                break
            if not line == ' ':
                line = line.split(' ')
                src = int(line[0])
                dst = int(line[1])
                value = float(line[2])
                temp = str(src) + '-' + str(dst)
                if not temp in OD_list:
                    continue
                data = str(src) + ' ' + str(dst) + ' ' + str(value) + '/n'
                f.write(data)

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
    plt.xlim((11, 100))
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
                  "GRU_KEY_CORRECT": "RED"}
    df = pd.read_csv(file_name)
    plt.xlim((0, 1))
    plt.xlabel("bias of maximum link utilization")
    plt.ylim((0,1))
    plt.ylabel("CDF")

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


    for i in range(len(labels)):
        if labels[i] == "CNN_LSTM":
            labels[i] = "LRCN"
        if "_OD_pair" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-OD"
        if "_KEY_CORRECT" in labels[i]:
            labels[i] = labels[i].split('_')[0] + "-KEC"

    plt.legend(labels, loc='lower right')
    # plt.savefig("MCF_MLU_bias_diff_model.png", bbox_inches='tight')
    plt.savefig("MCF_MLU_bias_diff_pers.png", bbox_inches='tight')
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
        LSTM_OD_TM_path = PATH_Abilene + "LSTM_OD_pair/LSTM_OD_pair_" + str(i + 1) + ".txt"
        GRU_OD_TM_path = PATH_Abilene + "GRU_OD_pair/GRU_OD_pair_" + str(i + 1) + ".txt"

        origin_TM.append(get_TM_volume(origin_TM_path))
        LSTM_OD_TM.append(get_TM_volume(LSTM_OD_TM_path))
        GRU_OD_TM.append(get_TM_volume(GRU_OD_TM_path))

    '''
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
    '''

    # 观察训练集的TM volume曲线，看看如何选择volume阈值
    file_name = "../OD_pair/Abilene-OD_pair_2004-08-01.csv"
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



if __name__ == "__main__":
    topo_file = "abilene.txt"
    out_file = "SDN_split_MLU_Abilene_EKM.csv"
    # strategy_path = [STRATEGY_ORIGIN, STRATEGY_LSTM, STRATEGY_LSTM_OD, STRATEGY_LSTM_KEY_correct,
    #                  STRATEGY_GRU, STRATEGY_GRU_OD, STRATEGY_GRU_KEY_CORRECT]
    # strategy_path = [STRATEGY_ORIGIN]
    strategy_path = [STRATEGY_ORIGIN, STRATEGY_GRU_EKM_OD, STRATEGY_GRU_EKM_KEC_7,
                     STRATEGY_GRU_EKM_KEC_14, STRATEGY_GRU_EKM_KEC_28, STRATEGY_GRU_EKM_KEC_43]
    nums = 1276
    first_line = ["Data"]
    for path in strategy_path:
        first_line.append(path.split("/")[-2])
    write_row_to_csv(first_line, out_file)

    # using strategy_tm to calculate strategy, and routing target_tm based on strategy
    result_list = []
    for i in range(len(strategy_path)):
        result_list.append([])

    for i in range(nums):
        print("Solving strategy for TM", i + 1)
        count = 0
        for path in strategy_path:
            G = construct_graph("Abilene.txt")
            strategy_file = path + path.split("/")[-2] + '_' + str(i + 1) + ".txt"
            strategy_tm_file = PATH_Abilene + path.split("/")[-2] + "/" + path.split("/")[-2] + "_" \
                               + str(i + 1) + ".txt"  # should be predicted TM file
            target_tm_file = Origin_TM + "Origin_" + str(i + 1) + ".txt"  # should be origin TM file
            # print(strategy_tm_file, strategy_file, target_tm_file)
            strategy = get_routing_strategy(strategy_tm_file, strategy_file)
            MLU = routing(G, target_tm_file, strategy, out_file)
            result_list[count].append(MLU)
            count += 1


    for i in range(nums):
        data = [i + 1]
        for j in range(len(result_list)):
            data.append(result_list[j][i])
        write_row_to_csv(data, out_file)

    '''
    MAE_result = ["MAE", "-"]
    RMSE_result = ["RMSE", "-"]
    for i in range(1, len(result_list)):
        MAE, RMSE = get_MAE_RMSE(result_list[0], result_list[i], nums)
        MAE_result.append(MAE)
        RMSE_result.append(RMSE)
    out_file = "SDN_split_MLU_MAE_RMSE_EKM.csv"
    write_row_to_csv(MAE_result, out_file)
    write_row_to_csv(RMSE_result, out_file)
    


    out_file = "SDN_split_MLU_bias_Abilene_EKM.csv"
    first_line = ["Data"]
    for path in strategy_path:
        first_line.append(path.split("/")[-2])
    write_row_to_csv(first_line, out_file)

    for i in range(nums):
        data = [i + 1, 0]
        for j in range(1, len(result_list)):
            data.append(abs(result_list[j][i] - result_list[0][i]) / result_list[0][i])
        write_row_to_csv(data, out_file)
    '''

    # draw cdf
    # draw_cdf("SDN_split_MLU_Abilene.csv", "MCF_MLU_result.png")
    # draw_cdf_bias("SDN_split_MLU_bias_Abilene.csv", "MCF_MLU_bias.png")
    # draw_MAE_RMSE("SDN_split_MLU_MAE_RMSE.csv", "MCF_MLU_MAE_RMSE.png")


    # analysis_overflow("overflow_SDN_split_MLU_Abilene.csv", 95, 1276)




    # G = construct_graph("Abilene.txt")
    # # get_OD_list("TM_Origin_1.txt", "TM_LSTM_1.txt")
    # strategy = get_routing_strategy("LSTM_KEY_CORRECT_220.txt", "flow_LSTM_KEY_CORRECT_220.txt")
    # routing(G, "Origin_220.txt", strategy, "out.txt")
    # print("=====================================================")
    # G = construct_graph("Abilene.txt")
    # strategy = get_routing_strategy("TM_Origin_577.txt", "flow_Origin_577.txt")
    # routing(G, "TM_Origin_577.txt", strategy, "out.txt")

    '''
    G = nx.DiGraph()
    node_list = []
    with open("test.txt", 'r') as readTxt:
        while True:
            line = readTxt.readline()
            if not line:
                break
            line = line.strip()
            line = line.split()
            src = int(line[2])
            dst = int(line[3])
            if not src in node_list:
                node_list.append(src)
                G.add_node(src)
            if not dst in node_list:
                node_list.append(dst)
                G.add_node(dst)
            G.add_edge(src, dst)

    
    print(list(nx.find_cycle(G, orientation='original')))
    for i in list(nx.find_cycle(G, orientation='original')):
        print(i[0], type(i))
    print(list(nx.all_simple_paths(G, source=10, target=11)))
    print(list(nx.simple_cycles(G)))
    '''
    '''
    G = nx.DiGraph()
    for i in range(6):
        G.add_node(i)
    G.add_edge(0, 3)
    G.add_edge(3, 2)
    G.add_edge(2, 1)
    G.add_edge(1, 0)
    G.add_edge(4, 5)
    G.add_edge(5, 4)
    print(list(nx.simple_cycles(G)))
    # print(list(nx.find_cycle(G, orientation='ignore')))
    '''
# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import numpy as np
import sys
import copy


def fileReader(fileName):
    """
    :param fileName:
    :return: data & column
    """
    data = []
    inpt = open(fileName)
    for i in inpt:
        data.append(i.strip('\n').split('\t'))
    matric = copy.deepcopy(np.array(data[1:]))
    col = copy.deepcopy(np.array(data[0]))
    return matric, col


def featureList(data_list, split_index):
    """
    select all the options in the attribute
    :param train_list:
    :param split_index:
    :return: a list of feature
    """
    return set(data_list[:, split_index])


def classify(data_list, split_index, feature_list):
    """
    :param data_list:
    :param split_index:
    :return: dict
    """
    count = dict()
    for i in feature_list:
        count[i] = data_list[data_list[:, split_index] == i]
    return count


def giniGain(file_input):
    gini_input, gini_col = fileReader(file_input)
    gini_index = len(gini_input[0])-1
    length = len(gini_input)
    attribute = featureList(gini_input, gini_index)
    """last index of D or R, unnecessary in benairy tree"""
    classify_result = classify(gini_input, gini_index, attribute)
    count_dic = {}
    p = {}
    imp = 0
    for i in attribute:
        count_dic[i] = np.unique(classify_result[i][:, -1], return_counts=True)[1][0]
        # print(count_dic)
        p[i] = float(count_dic[i]/length)
        imp +=p[i]*p[i]
    count_array = []
    for j in attribute:
        count_array.append(count_dic[j])
    majority_index = np.argmax(count_array)
    error = 1 - float(count_array[majority_index]/length)
    return 1-imp, error

if __name__ == '__main__':
    # TRAIN_INPUT = "small_train.tsv"
    # INSPECT_OUTPUT = "small_inspect.txt"
    TRAIN_INPUT = sys.argv[1]
    INSPECT_OUTPUT = sys.argv[2]
    gini_impurity, gini_error = giniGain(TRAIN_INPUT)

    print(gini_impurity, gini_error)
    with open(INSPECT_OUTPUT, 'w') as mof:
         mof.write('gini_impurity: ' + str(gini_impurity) + '\n')
         mof.write('error: ' + str(gini_error) + '\n')

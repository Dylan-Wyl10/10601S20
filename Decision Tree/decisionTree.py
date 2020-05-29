# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import numpy as np
import sys
import copy
from collections import Counter


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

def errorRate(result_predict, result_real):
    error_num = np.count_nonzero(result_real != result_predict)
    return error_num / len(result_predict)


def classify(data_list, split_index):
    """
    :param data_list:
    :param split_index:
    :return: Counter
    """
    return Counter(data_list[:, split_index])


def probaInList(X, data_list):
    """
    :param X:
    :param List:
    :return: P(X)in List
    """
    length = len(data_list)
    countX = Counter(data_list)
    proba = float(countX[X] / length)
    return proba


def attributes(data_list):
    return np.unique(data_list)
    """去除重复值"""


def giniImpurity(list_label):
    label_attri = np.unique(list_label)
    l = len(list_label)
    label_dict = dict(Counter(list_label))
    p = {}
    imp = 0
    for i in label_attri:
        p[i] = float(label_dict[i]/l)
        imp += p[i] * p[i]
    return 1-imp


def giniGain(list_input, list_label):
    """
    :param list_input: 数据集中某一列
    :param list_label: [:,-1]
    :return: gini gain
    """
    list_Len = len(list_input)
    count_dict = dict(Counter(list_input))
    list_attribute = attributes(list_input)
    gini = giniImpurity(list_label)
    for i in list_attribute:
        gini -= (count_dict[i]/list_Len)*giniImpurity(list_label[list_input == i])
    return gini


def majorityVote(list_label):
    count_dic = dict(Counter(list_label))
    labels, counts = np.unique(list_label, return_counts=True)
    if len(labels) == 1:
        return labels[0]
    if counts[0] == counts[1]:
        return np.sort(labels)[1]
    a = 0
    b = None
    for k, v in count_dic.items():
        if v > a:
            a = v
            b = k
    return b


class trNode():
    def __init__(self, data_matric, data_label, tree_depth, col_name, restAttri_index, max_depth, upper_Node = None):
        """
        :param data_matric:
        :param data_label:
        :param tree_depth:
        :param col_name:
        :param restAttri_index:
        :param max_depth:
        :param upper_Node:
        """
        self.sons = [None, None]
        self.son_labels = [None, None]
        self.upper_Node = upper_Node
        self.data_matric = data_matric
        self.data_label = data_label
        self.label = majorityVote(data_label)
        self.tree_depth = tree_depth
        self.col_name = col_name
        self.dataId_list = restAttri_index  # int number, recording the attribute index
        self.max_depth = max_depth

    def selectAttribute(self):
        """
        :return: maximize gini gain attribute for this layer in the tree， int num
        """
        gini_list = []
        # print(self.dataId_list)
        for i in self.dataId_list:
            gini_list.append(giniGain(self.data_matric[:, i], self.data_label))
        print(gini_list)
        if np.max(gini_list) == 0:
            return -1
        maxgini_index = np.argmax(gini_list)
        print(maxgini_index)
        print(self.dataId_list[maxgini_index])
        return self.dataId_list[maxgini_index]

    def growTree(self):
        self.split_index = self.selectAttribute()# int number, split according to this number in this node
        # print(split_index)
        if self.split_index == -1:
            return
        self.col_id = self.col_name[self.split_index]
        split_attr = self.data_matric[:, self.split_index]#['n','n'...'y','y']
        split_feature = attributes(split_attr)#['y','n']
        new_colname = copy.deepcopy(self.col_name)
        print(split_feature)
        print(new_colname)
        print(self.label)
        new_Idlist = copy.deepcopy(self.dataId_list)
        new_Idlist.remove(self.split_index)
        print(self.col_name[new_Idlist])
        if len(split_feature) == 2:
            # if (split_attr == split_feature[0]).any():
                self.sons[0] = trNode(self.data_matric[split_attr == split_feature[0], :],
                               self.data_label[split_attr == split_feature[0]],
                               self.tree_depth + 1, new_colname, new_Idlist, self.max_depth, self)
                self.son_labels[0] = split_feature[0]
                # print('yes,0')
            # if (split_attr == split_feature[1]).any():
                self.sons[1] = trNode(self.data_matric[split_attr == split_feature[1], :],
                               self.data_label[split_attr == split_feature[1]],
                               self.tree_depth + 1, new_colname, new_Idlist, self.max_depth, self)
                self.son_labels[1] = split_feature[1]
                # print('yes,1')

        elif len(split_feature) == 1:
            self.sons[0] = trNode(self.data_matric[split_attr == split_feature[0], :],
                                self.data_label[split_attr == split_feature[0]],
                                self.tree_depth + 1, new_colname, new_Idlist, self.max_depth, self)
            self.son_labels[0] = split_feature[0]

            # self.sons[1] = None
            # self.son_labels[1] = None

            # print('yes,2')

    def train(self):
        if self.tree_depth == self.max_depth:
            return
        if len(attributes(self.data_label)) == 1:# only 1 label, perfect node, stop
            # # print(len(attributes(self.label)))
            # # print(attributes(self.label))
            # print('www')
            return
        if self == None:
            return
        self.growTree()
        if self.sons[0]:
            self.sons[0].train()
        if self.sons[1]:
            self.sons[1].train()

    def readNode(self, test_list):
        if self.sons[0] is None and self.sons[1] is None:
            return self.label
        elif self.son_labels[0] == test_list[self.split_index]:
            return self.sons[0].readNode(test_list)
        elif self.son_labels[1] == test_list[self.split_index]:
            return self.sons[1].readNode(test_list)
        else:
            return self.label

    def printNode(self, pos=None):
        print_label_list = np.unique(self.data_label)
        print_label_dict = dict(Counter(self.data_label))
        # print(print_label_dict)
        # print(print_label_list)
        # cnum = np.array([np.count_nonzero(self.data_label == i) for i in cate])
        if self.tree_depth == 1:
            print('[', end='')
            for i in range(len(print_label_list)):
                print(print_label_list[i] + str(print_label_dict[print_label_list[i]])+'/', end='')
            print(']')
        else:
            for i in range(self.tree_depth - 1):
                print('|\t', end='')
            label = self.upper_Node.son_labels[0] if pos == 'sons0' else self.upper_Node.son_labels[1]
            print(self.col_name[self.upper_Node.split_index] + ' = ' + str(label) + ': ', end='')
            for i in range(len(print_label_list)):
                print(print_label_list[i] + ' ' + str(print_label_dict[print_label_list[i]]), end='')
                print('/', end='')
            print(']', end=' ')
            print(self.label)
        return

    def printTree(self, pos=None):
        if self.max_depth == 1:
            self.printNode()
        else:
            self.printNode(pos)
            if self.sons[0]:
                self.sons[0].printTree('sons0')
            if self.sons[1]:
                self.sons[1].printTree('sons1')
        return


class DecisionTree:
    def __init__(self, max_depth, data_file, input_matric):
        self.max_depth = max_depth
        self.data_file = data_file
        self.tree_data, self.tree_col = fileReader(self.data_file)
        self.tree_matric = self.tree_data[:, :-1]
        self.tree_label = self.tree_data[:, -1]
        if input_matric is None:
            self.input_matric = self. tree_matric
        else:
            self.input_matric = input_matric
        index_id = []
        for i in range(self.tree_data[:-1].shape[1] - 1):
            index_id.append(i)
        self.root = trNode(self.tree_matric, self.tree_label, 1, self.tree_col, index_id, self.max_depth)
        self.root.train()
        # print('1')

    def readTree(self):
        label_output = []
        for test in self.input_matric:
            label_output.append(self.root.readNode(test))
        np.array(label_output)
        self.result = label_output
        return

    def printDT(self):
        self.root.printTree()


if __name__ == '__main__':
    # TRAIN_INPUT = "politicians_train.tsv"
    # TEST_INPUT = "politicians_test.tsv"
    # METRIC_OUTPUT = 'politicians_metric.txt'
    # MAX_DEPTH = int(6)
    TRAIN_INPUT = sys.argv[1]
    TEST_INPUT = sys.argv[2]
    MAX_DEPTH = int(sys.argv[3])
    TRAIN_OUTPUT = sys.argv[4]
    TEST_OUTPUT = sys.argv[5]
    METRIC_OUTPUT = sys.argv[6]
    MAX_DEPTH1 = MAX_DEPTH + 1

    ipt_train, col_train = fileReader(TRAIN_INPUT)
    ipt_test, col_test = fileReader(TEST_INPUT)
    Tree1 = DecisionTree(MAX_DEPTH1, TRAIN_INPUT, ipt_train[:, :-1])
    Tree1.readTree()
    Tree2 = DecisionTree(MAX_DEPTH1, TRAIN_INPUT, ipt_test[:, :-1])
    Tree2.readTree()
    Tree1.printDT()
    error_train = errorRate(Tree1.result, ipt_train[:, -1])
    error_test = errorRate(Tree2.result, ipt_test[:, -1])
    print(error_train)
    print(error_test)

    with open(TRAIN_OUTPUT, 'w') as nof:
        for i in Tree1.result:
            nof.write(str(i) + '\n')
    with open(TEST_OUTPUT, 'w') as sof:
        for i in Tree2.result:
            sof.write(str(i) + '\n')
    with open(METRIC_OUTPUT, 'w') as mof:
         mof.write('error(train) ' + str(error_train) + '\n')
         mof.write('error(test): ' + str(error_test) + '\n')

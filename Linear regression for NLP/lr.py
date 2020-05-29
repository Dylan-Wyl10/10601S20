# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import numpy as np
import sys


def processing_split(file, n):
    # 分离x,y 两个label
    ipt = open(file)
    x = []
    y = []
    for i in ipt:
        li = i.split('\t')
        y.append(int(li[0]))
        line = {}
        for ele in li[1:]:
            line[int(ele.split(":")[0])] = int(ele.split(":")[1])
            line[n] = 1
        x.append(line)
    return x, y # x= dict list, y = 1d list


def creat_theta_list(file):
    ipt = open(file)
    count = 0
    for ii in ipt:
        count += 1
    # print(count)
    return [0]*(count+1), count


class Lr():
    def __init__(self, data_file, dict_file, opt_file):
        self.theta_list, self.lines_count = creat_theta_list(dict_file)
        self.x, self.y = processing_split(data_file, self.lines_count)
        self.opt_file = opt_file

    def sgd(self, yi, x_line_dict):
        yeta = 0.1
        theta_T_x = 0
        for key, val in x_line_dict.items():
            theta_T_x += val * self.theta_list[key]
        exp_thetaTx = np.exp(theta_T_x)
        self.sigema = exp_thetaTx/(1+exp_thetaTx)
        v = (yi - self.sigema) * yeta
        for key, val in x_line_dict.items():
            self.theta_list[key] += val * v
        # print(self.theta_list)

    def train(self, epoch):
        for i in range(epoch):
            for count in range(len(self.y)):
                self.sgd(self.y[count], self.x[count])
            print(i)
        return self.theta_list

    def predict(self, theta_list):
        self.y_pre = []
        opt = open(self.opt_file, 'w')
        for count in range(len(self.y)):
            theta_T_x1 = 0
            for key, val in self.x[count].items():
                theta_T_x1 += val * theta_list[key]
            exp_thetaTx1 = np.exp(theta_T_x1)
            sigema1 = exp_thetaTx1/(1+exp_thetaTx1)
            if sigema1 > 0.5:
                self.y_pre.append(1)
                opt.write('1' + '\n')
            else:
                self.y_pre.append(0)
                opt.write('0' + '\n')
            # print(self.y_pre)

    def error_rate(self):
        error = 0
        for i in range(len(self.y)):
            if self.y[i] != self.y_pre[i]:
                print('warning')
                error += 1
        return error/len(self.y)


if __name__ == "__main__":

    TRAIN_IPT = sys.argv[1]
    VALID_IPT = sys.argv[2]
    TEST_IPT = sys.argv[3]
    DICT = sys.argv[4]
    TRAIN_OPT = sys.argv[5]
    TEST_OPT = sys.argv[6]
    METRIC_OPT = sys.argv[7]
    EPOCH = int(sys.argv[8])

    Lr_train = Lr(TRAIN_IPT, DICT, TRAIN_OPT)
    list1 = Lr_train.train(EPOCH)
    Lr_train.predict(list1)

    Lr_test = Lr(TEST_IPT, DICT, TEST_OPT)
    Lr_test.predict(list1)

    error_train = Lr_train.error_rate()
    error_test = Lr_test.error_rate()

    print(error_test, error_train)

    with open(METRIC_OPT, 'w') as mof:
        mof.write('error(train): ' + str(error_train) + '\n')
        mof.write('error(test): ' + str(error_test) + '\n')
    #
    # ll = Lr("model1_formatted_train1.tsv", "dict.txt", "out.labels")
    # list1 = ll.train(30)
    # print(list1)
    # ll.predict(list1)
    #
    # lll = Lr("model1_formatted_test1.tsv", "dict.txt", "out.labels")
    # lll.predict(list1)
    # print(ll.error_rate())
    # print(lll.error_rate())

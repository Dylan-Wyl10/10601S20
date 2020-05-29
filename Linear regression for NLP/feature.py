# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import sys
from collections import Counter


def read_and_split(file):
    f = open(file)
    word_dict = {}
    for i in f:
        word_list = i.strip('\n').split()
        word_dict[word_list[0]] = word_list[1]
    return word_dict


def model(file_input, file_output, model_idx, word_dict):
    ipt = open(file_input)
    opt = open(file_output, 'w')
    label_list = []
    for i in ipt:
        label_list.append(i[0])
        opt_line = i[0]
        word_list = i[2:].split()
        new_word_list = []
        for j in word_list:
            if j in word_dict.keys():
                idx = word_dict[j]
                new_word_list.append(idx)
        li = dict(Counter(new_word_list))
        for key, val in li.items():
            if model_idx == 1:
                opt_line += '\t' + key + ':' + '1'
            elif model_idx == 2:
                if val < 4:
                    opt_line += '\t' + key + ':' + '1'
        opt_line += '\n'
        opt.write(opt_line)


if __name__ == '__main__':
    TRAIN_IPT = sys.argv[1]
    VALID_IPT = sys.argv[2]
    TEST_IPT = sys.argv[3]
    DICT_IPT = sys.argv[4]
    TRAIN_OPT = sys.argv[5]
    VALID_OPT = sys.argv[6]
    TEST_OPT = sys.argv[7]
    MODEL_IDX = int(sys.argv[8])
    # TRAIN_IPT = "train_data.tsv"
    # VALID_IPT = "valid_data.tsv"
    # DICT_IPT = "dict.txt"
    # TRAIN_OPT = "output_train.tsv"
    # VALID_OPT = "output_valid.tsv"
    # MODEL_IDX = 1
    word_dict = read_and_split(DICT_IPT)
    model(TRAIN_IPT, TRAIN_OPT, MODEL_IDX, word_dict)
    model(VALID_IPT, VALID_OPT, MODEL_IDX, word_dict)
    model(TEST_IPT, TEST_OPT, MODEL_IDX, word_dict)

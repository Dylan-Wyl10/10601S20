
import numpy as np
import sys
import copy
from collections import Counter


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
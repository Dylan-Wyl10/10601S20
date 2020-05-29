# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import numpy as np
import sys


def splitWords(file):
    f = open(file)
    word_matrix, tag_matrix = [], []
    for i in f:
        temp = i.strip('\n').split(' ')
        wli = [] #word_list for lines
        tli = [] #tag_list for lines
        for ii in temp:
            wli.append(ii.split('_')[0])
            tli.append(ii.split('_')[1])
        word_matrix.append(wli)
        tag_matrix.append(tli)
    return word_matrix, tag_matrix #output two split 2darray


def splitIndex(file, mod):
    f = open(file)
    if mod == 1:
        idxli = {}
        idx = 0
        for i in f:
            idxli[i.strip('\n')] = idx
            idx += 1
    else:
        idxli = []
        for i in f:
            idxli.append(i.strip('\n'))
    return idxli


def getParameter(file):
    f = open(file)
    tem = []
    for i in f:
        tem.append(i.strip('\n').split(' '))
    tem = np.array(tem, dtype=float)
    return tem


def predictforward(a, b, pi, wdidx, wdline):
    alpha = np.zeros((a.shape[0], len(wdline)))
    alpha[:, 0] = pi[:, 0] * b[:, wdidx[wdline[0]]]
    for t in range(1, len(wdline)):
        for j in range(a.shape[0]):
            temp_alpha = []
            for k in range(a.shape[0]):
                temp_alpha.append(alpha[k, t-1] * a[k, j])
            alpha[j, t] = b[j, wdidx[wdline[t]]] * np.sum(temp_alpha)
    # print(alpha)
    return alpha


def predictbackward(a, b, wdidx, wdline):
    beta = np.zeros((a.shape[0], len(wdline)))
    beta[:, -1] = 1
    for t in range(len(wdline)-2, -1, -1):
        for j in range(a.shape[0]):
            sum = beta[j, t]
            for k in range(a.shape[0]):
                sum += b[k, wdidx[wdline[t+1]]] * beta[k, t+1] * a[j, k]
            beta[j, t] = sum
    return beta


def predict(wdli, wdidx, tgidx):
    predli,logli = [], []
    for line in range(len(wdli)):
        temp = []
        wdline = wdli[line]
        alp = predictforward(a, b, pi, wdidx, wdline)
        bta = predictbackward(a, b, wdidx, wdline)
        print(bta)
        for j in range(alp.shape[1]):
            predidx = np.argmax(alp[:, j] * bta[:, j])
            # print(alp[:, j], bta[:, j])
            # print(predidx)
            temp.append(tgidx[predidx])
            # print(temp)
        predli.append(temp)
        llh = np.log(np.sum(alp[:, -1]))
        logli.append(llh)
        # print(logli)
    return np.sum(logli)/len(wdli), predli


def writeprediction(wdli, predli):
    result = []
    for row in range(len(wdli)):
        tem = []
        for col in range(len(wdli[row])):
            point = str(wdli[row][col]) + '_' + str(predli[row][col])
            tem.append(point)
        result.append(tem)
    return result


def errorcal(predli, realli):
    count = 0
    size = 0
    for i in range(len(predli)):
        for j in range(len(predli[i])):
            count += (predli[i][j] == realli[i][j])
            size += 1
    print(count)
    return count/size


def writefiles(prediction, matrix, loglik, result, error):
    # print(type(result))
    with open(prediction, "w") as pred:
        for i in range(len(result)):
            for j in range(len(result[i]) - 1):
                pred.write(str(result[i][j]) + ' ')
            pred.write(str(result[i][-1]) + '\n')
    with open(matrix, 'w') as matri:
        # matri.write('Average Log-Likelihood: ')
        matri.write('Average Log-Likelihood: ' + str(loglik) + '\n' + 'Accuracy: '
                    + str(error) + '\n')


if __name__ == '__main__':
    # Train_Input = sys.argv[1]
    # Index_2_Word = sys.argv[2]
    # Index_2_Tag = sys.argv[3]
    # Hmmprior = sys.argv[4]
    # Hmmemit = sys.argv[5]
    # Hmmtrans = sys.argv[6]
    # Prediction = sys.argv[7]
    # Matrix = sys.argv[8]

    Train_Input = 'toy_data/toytest.txt'
    Index_2_Word = 'toy_data/toy_index_to_word.txt'
    Index_2_Tag = 'toy_data/toy_index_to_tag.txt'
    Hmmprior = 'toy_data/toy_hmmprior.txt'
    Hmmemit = 'toy_data/toy_hmmemit.txt'
    Hmmtrans = 'toy_data/toy_hmmtrans.txt'
    Prediction = 'toy_data/toy_predicttest.txt'
    Matrix = 'toy_data/toy_metrics.txt'

    # Train_Input = 'testwords.txt'
    # Index_2_Word = 'index_to_word.txt'
    # Index_2_Tag = 'index_to_tag.txt'
    # Hmmprior = 'hmmprior.txt'
    # Hmmemit = 'hmmemit.txt'
    # Hmmtrans = 'hmmtrans.txt'
    # Prediction = 'predicttest11.txt'
    # Matrix = 'metrics11.txt'

    wdli, tgli = splitWords(Train_Input)
    wdidx = splitIndex(Index_2_Word, 1)
    tgidx = splitIndex(Index_2_Tag, 2)
    pi = getParameter(Hmmprior)
    a = getParameter(Hmmtrans)
    b = getParameter(Hmmemit)

    log, pred_list = predict(wdli, wdidx, tgidx)

    results = writeprediction(wdli, pred_list)

    acc = errorcal(pred_list, tgli)

    writefiles(Prediction, Matrix, log, results, acc)





import math

def Evaluation(labels, predicts):
    TP, TN, FN, FP = 0, 0, 0, 0

    for i, j in zip(labels, predicts):
        if j == i and i == 1:
            TP += 1
        elif j == i and i == 0:
            TN += 1
        elif j != i and i == 1:
            FN += 1
        elif j != i and i == 0:
            FP += 1

    if TP + TN + FN + FP != 0:
        ACC = (TP + TN) / (TP + TN + FN + FP)
    else:
        ACC = 0
    if TP + FP != 0:
        PRE = TP / (TP + FP)
    else:
        PRE = 0
    if TP + FN != 0:
        SN = TP / (TP + FN)
    else:
        SN = 0
    if TN + FP != 0:
        SP = TN / (TN + FP)
    else:
        SP = 0
    if PRE + SN != 0:
        F_score = 2 * PRE * SN / (PRE + SN)
    else:
        F_score = 0
    # MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    return ACC, PRE, SN, SP, F_score


def calcAUC(labels, probs):
    N = 0
    P = 0
    neg_prob = []
    pos_prob = []
    for _,i in enumerate(labels):
        if (i == 1):
            P += 1
            pos_prob.append(probs[_][1])
        else:
            N += 1
            neg_prob.append(probs[_][0])
    number = 0
    for pos in pos_prob:
        for neg in neg_prob:
            if (pos > neg):
                number += 1
            elif (pos == neg):
                number += 0.5
    return number / (N * P)

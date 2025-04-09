from sklearn import metrics
import numpy as np
import torch


def evaluation(score, label):

    microAP = metrics.average_precision_score(label, score, average="micro")
    microAUC = metrics.roc_auc_score(label, score, multi_class="ovr", average="micro")
    macroAP = metrics.average_precision_score(label, score, average="macro")
    macroAUC = metrics.roc_auc_score(label, score, multi_class="ovr", average="macro")

    eval_result = (microAP + macroAUC + microAUC + macroAP) / 4
    return [
        macroAP * 100,
        macroAUC * 100,
        microAP * 100,
        microAUC * 100,
        eval_result * 100,
    ]

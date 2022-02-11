import numpy as np
from sklearn.metrics import confusion_matrix


def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    n = labels.shape[0]
    fprs, tprs = np.zeros(n), np.zeros(n)
    thrs = np.linspace(0, 1, 50)
    for i, t in enumerate(thrs):
        predicted = probabilities >= t
        tn, fp, fn, tp = confusion_matrix(labels, predicted).flatten()
        tprs[i] = tp / (tp + fn)
        fprs[i] = fp / (fp + tn)
    return fprs, tprs, thrs

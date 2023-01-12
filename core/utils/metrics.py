from sklearn.metrics import confusion_matrix

def calc_detection_performance(y_true, y_pred):
    """
    calculate anomaly detection performance

    Parameters
    ----------
    y_true : ndarray or list
        The ground truth labels
    y_pred : ndarray or list
        The predicted labels
    
    Returns
    -------
    list 
        list for results, [f1,precision,recall,TP,TN,FP,FN]
    """

    TN,FP,FN,TP = confusion_matrix(y_true,y_pred).ravel()
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return [f1,precision,recall,TP, TN, FP, FN]


    
'''
Created on Nov 9, 2022

@author: z003w5we
'''

import sys,getopt
sys.path.insert(0, "../")
from core.preprocessing.data_loader import DATASET,load_dataset
from core.preprocessing import DataUtil
from sklearn.metrics import roc_auc_score
from core.model.reconstruction_models.DeepSVDD import DeepSVDD


if __name__ == "__main__":
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "d:",["dataset="])
    except:
        print(u'python main_deepsvdd.py --dataset <dataset>')
        sys.exit(2)
    
    
    ds = None
    gamma = None
    theta = None
    load_saved = False
    
    for opt, arg in opts:
        if opt in ['-d','--dataset']:
            if arg.lower() == 'swat':
                ds = DATASET.SWAT
            elif arg.lower() == 'batadal':
                ds = DATASET.BATADAL
            elif arg.lower() == 'kddcup99':
                ds = DATASET.KDDCup99
            elif arg.lower() == 'gaspipeline':
                ds = DATASET.GasPipeline
            elif arg.lower() == 'annthyroid':
                ds = DATASET.Annthyroid
            elif arg.lower() == 'cardio':
                ds = DATASET.Cardio
        else:
            print(u'python main_deepsvdd.py --dataset <dataset>')
            sys.exit(2)
    
    if ds is None:
        print(u'please specify dataset!')
        sys.exit(2)
    
    print(ds)

    train_df, test_df, test_labels, signals = load_dataset(ds)
    du = DataUtil(signals, scaling_method='min_max')
    train_df = du.normalize_and_encode(train_df)
    test_df = du.normalize_and_encode(test_df)

    model = DeepSVDD(signals)
    model.train(train_df.values, z_dim=len(signals)//4+1, nu=0.01, hidden_layers=2, z_activation='tanh', batch_size=256,epochs=100, verbose=0)

    scores = model.score_samples(test_df.values)
    auc = roc_auc_score(test_labels, scores)
    pauc = roc_auc_score(test_labels, scores, max_fpr=0.1)
    print(ds)
    print('AUC:',auc)
    print('pAUC',pauc)
    print()


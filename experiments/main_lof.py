import sys,getopt
sys.path.insert(0, "../")

from core.preprocessing.data_loader import load_dataset,DATASET
from sklearn.neighbors import LocalOutlierFactor
from core.preprocessing import DataUtil
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "d:",["dataset="])
    except:
        print(u'python main_lof.py --dataset <dataset>')
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
            print(u'python main_lof.py --dataset <dataset>')
            sys.exit(2)
    
    if ds is None:
        print(u'please specify dataset!')
        sys.exit(2)
        
    print(ds)
    train_df,test_df,test_label,signals = load_dataset(ds)
    
    du = DataUtil(signals,scaling_method='min_max')
    train_df = du.normalize_and_encode(train_df)
    test_df = du.normalize_and_encode(test_df)
    
    model = LocalOutlierFactor(novelty=True)
    model.fit(train_df.values)
    
    true_labels = test_label
    
    scores = model.decision_function(test_df.values)
    scores = -1*scores
    
    auc_roc = roc_auc_score(true_labels,scores)
    pauc = roc_auc_score(true_labels,scores,max_fpr=0.1)
    print('auc',auc_roc)
    print('pauc',pauc)
    print()
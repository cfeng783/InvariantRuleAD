import sys,getopt
sys.path.insert(0, "../")
from core.model.reconstruction_models import Autoencoder
from core.preprocessing.data_loader import DATASET,load_dataset
from core.learning.hp_optimization.Hyperparameter import UniformIntegerHyperparameter,ConstHyperparameter,CategoricalHyperparameter
from core.learning.hp_optimization import HPOptimizers
from core.preprocessing import DataUtil
from core.preprocessing import train_val_split
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "d:",["dataset="])
    except:
        print(u'python main_ae.py --dataset <dataset>')
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
            print(u'python main_ae.py --dataset <dataset>')
            sys.exit(2)
    
    if ds is None:
        print(u'please specify dataset!')
        sys.exit(2)
    
    print(ds)
    train_df,test_df,test_label,signals = load_dataset(ds)
    
    data_util = DataUtil(signals,scaling_method='min_max')
    train_df = data_util.normalize_and_encode(train_df)
    test_df = data_util.normalize_and_encode(test_df)
    
    train_df, val_df = train_val_split(train_df,val_ratio=0.2)
    
    hidden_dim = len(signals)//4
    hp_list = []
    hp_list.append(ConstHyperparameter('hidden_dim',hidden_dim)) 
    hp_list.append(UniformIntegerHyperparameter('num_hidden_layers',1,3))
    hp_list.append(ConstHyperparameter('save_best_only',True))
    hp_list.append(UniformIntegerHyperparameter('epochs',50,100))
    hp_list.append(ConstHyperparameter('verbose',2))
    hp_list.append(CategoricalHyperparameter('batch_size',[64,256]))
    
    model = Autoencoder(signals)
    optor = HPOptimizers.RandomizedGridSearch(model, hp_list,train_df.values, val_df.values)
    optModel,optHPCfg,bestScore = optor.run(n_searches=2,verbose=0)
    # print('optHPCfg',optHPCfg)
    # print('bestScore',bestScore)
    
    scores = optModel.score_samples(test_df.values,return_evidence=False)
    labels = optModel.data_handler.extract_labels(test_label)
    
    auc_roc = roc_auc_score(labels,scores)
    pauc = roc_auc_score(labels,scores,max_fpr=0.1)
    print('auc',auc_roc)
    print('pauc',pauc)
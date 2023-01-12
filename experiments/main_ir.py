import sys,getopt
sys.path.insert(0, "../")

from core.preprocessing.data_loader import load_dataset,DATASET
from core.preprocessing import DataUtil
from core.model.rule_models.invariant_model import InvariantRuleModel,PredicateMode
from sklearn.metrics import roc_auc_score
from core.preprocessing.signals import ContinuousSignal
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

 

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "d:m:t:g:r",["dataset=","mode=","theta=","gamma=","reproduce"])
    except:
        print(u'python main_ir.py --dataset <dataset> --mode <mode> --theta <theta> --gamma <gamma> --reproduce')
        sys.exit(2)
    
    
    ds = None
    gamma = None
    theta = None
    load_saved = False
    mode = PredicateMode.DTImpurity
    
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
        elif opt in ['-m','--mode']:
            if arg.lower() == 'uniformbins':
                mode = PredicateMode.UniformBins
            elif arg.lower() == 'kmeansbins':
                mode = PredicateMode.KMeansBins
            elif arg.lower() == 'dtimpurity':
                mode = PredicateMode.DTImpurity
            else:
                print(u'please specify right mode')
                sys.exit(2)
        elif opt in ['-t','--theta']:
            theta = float(arg)
        elif opt in ['-g','--gamma']:
            gamma = float(arg)
        elif opt in ['-r','--reproduce']:
            load_saved = True
        else:
            print(u'python main_ir.py --dataset <dataset> --mode <mode> --theta <theta> --gamma <gamma> --reproduce')
            sys.exit(2)
    
    if ds is None:
        print(u'please specify dataset!')
        sys.exit(2)
        
    if load_saved == False:
        if theta is None or gamma is None:
            print(u'please specify theta and gamma!')
            sys.exit(2)
    
    print(ds)
    train_df,test_df,test_labels,signals = load_dataset(ds)
    print('train size',len(train_df))
    print('test size',len(test_df))
    num_cont, num_cate = 0, 0
    for signal in signals:
        if isinstance(signal, ContinuousSignal):
            num_cont += 1
        else:
            num_cate += 1
    print(f'number of continuous signals: {num_cont}')
    print(f'number of categorical signals: {num_cate}')
    print('anomaly ratio:',list(test_labels).count(1)/len(test_labels))
    
    du = DataUtil(signals,scaling_method=None)
    train_df = du.normalize_and_encode(train_df)
    test_df = du.normalize_and_encode(test_df)
        
    irm = InvariantRuleModel(signals,mode)
    
    if load_saved == False:
        irm.train(train_df, max_predicates_per_rule=5, gamma=gamma, theta=theta,use_multicore=True)
        # irm.save_model('../results/'+str(mode), str(ds))
    else:
        irm.load_model('../results/'+str(mode), str(ds))
    
    num_rules = irm.get_num_rules()
    print('num rules',num_rules)
    
    anomaly_scores = irm.predict(test_df,use_boundary_rules=True,use_cores=1)
    
    auc  = roc_auc_score(test_labels[:len(anomaly_scores)],anomaly_scores)
    pauc = roc_auc_score(test_labels[:len(anomaly_scores)],anomaly_scores,max_fpr=0.1)
    print('AUC',auc)
    print('pAUC',pauc)
    print()
        
    # if ds == DATASET.SWAT:
    #     print('---check violated rules for anomaly segments in SWAT---')
    #     segments = []
    #     for i in range(1,len(test_labels)):
    #         if test_labels[i] == 1 and test_labels[i-1] == 0:
    #             sidx = i
    #         if test_labels[i] == 0 and test_labels[i-1] == 1:
    #             segments.append((sidx,i-1))
    #     print('num segs',len(segments))
    #     print(segments)
    #
    #     ret_json = {}
    #     ret_json['abnormal_segments'] = []
    #     for seg in segments:
    #         print(seg)
    #         seg_dict = {'start_index':seg[0], 'end_index':seg[1]}
    #         anomaly_df = test_df.loc[seg[0]:seg[1]+1,:]
    #         exp = irm.get_anomaly_explanation(anomaly_df, use_boundary_rules=True)
    #         causes = []
    #         for exp_item in exp.summary():
    #             causes.append( {'feature':exp_item[0],'probability':exp_item[1],'violated_rule':exp_item[2],'involved_features':exp_item[4],'violated_locations':exp_item[3]} ) 
    #         seg_dict['causes'] = causes
    #         ret_json['abnormal_segments'].append(seg_dict)
    #         # print(exp.summary())
    #         # print()
    #     with open("../results/abnormal_segments.json", "w") as outfile:
    #         json.dump(ret_json, outfile,cls=NpEncoder)
    #     print('violated rules saved in ../results/abnormal_segments.json')
        
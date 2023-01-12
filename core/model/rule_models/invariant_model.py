from .. import BaseModel,AnomalyDetector
from ...utils import override
import random,pickle
from .rule_mining import RuleMiner
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
from ...preprocessing.signals import ContinuousSignal
import numpy as np
import multiprocessing
from . import helper
from .anomaly_explanation import AnomalyExplanation


from enum import Enum

class PredicateMode(Enum):
    UniformBins = 201
    KMeansBins = 202
    DTImpurity = 203

class InvariantRuleModel(BaseModel,AnomalyDetector):
    '''
    Invariant rule model for interpretable anomaly detection
    
    Parameters
    ----------
    signals : list
        the list of signals the model is dealing with
    '''
    def __init__(self, signals, mode=PredicateMode.DTImpurity):
        self._signals = signals
        self._mode = mode
        
        self._cont_feats = []
        self._disc_feats = []
        self._cont_signals = []
        self._disc_signals = []
        for signal in self._signals:
            if isinstance(signal, ContinuousSignal):
                self._cont_feats.append(signal.name)
                self._cont_signals.append(signal)
            else:
                self._disc_feats.extend(signal.get_onehot_feature_names())
                self._disc_signals.append(signal)
        
    
    
    def _propose_cutoff_values(self,df,min_samples_leaf,nbins=5):
        cutoffs = {}
        for signal in self._cont_signals:
            cutoffs[signal.name] = []
        
        if self._mode == PredicateMode.DTImpurity:
            for signal in self._disc_signals:
                if len(self._cont_feats) > 0:
                    onehot_feats = signal.get_onehot_feature_names()
                    df.loc[:,'tempt_label'] = 0
                    for i in range(len(onehot_feats)):
                        df.loc[df[onehot_feats[i]]==1,'tempt_label'] = i
    
                    xfeats = list(self._cont_feats)
                    x = df[xfeats].values
                    y = df['tempt_label'].values
                    df.drop(columns='tempt_label',inplace=True)
                    model = DecisionTreeClassifier(criterion = "entropy",min_samples_leaf=int(min_samples_leaf))
                    model.fit(x,y)
                    cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
                    for ct in cut_tuples:
                        cutoffs[ct[0]].append( (ct[1],ct[2]) )
            
            for signal in self._cont_signals:
                yfeat = signal.name
                xfeats = list(self._cont_feats)
                xfeats.remove(yfeat)
            
                
                x = df[xfeats].values
                y = df[yfeat].values
                y = (y-signal.mean_value)/signal.std_value
                model = DecisionTreeRegressor(min_samples_leaf=int(min_samples_leaf))
            
                model.fit(x,y)
                cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
                for ct in cut_tuples:
                    cutoffs[ct[0]].append( (ct[1],ct[2]) )
        
        else:
            if self._mode == PredicateMode.KMeansBins:
                kbins = 'kmeans'
            elif self._mode == PredicateMode.UniformBins:
                kbins = 'uniform'
            for signal in self._cont_signals:
                y = df[signal.name].values
                discretizer = KBinsDiscretizer(n_bins=nbins,encode='ordinal',strategy=kbins)
                y = discretizer.fit_transform(np.reshape(y,(-1,1)))
                edges = discretizer.bin_edges_[0]
                for i in range(1,len(edges)-1):
                    cutoffs[signal.name].append( (edges[i],1) ) 
        
        cutoffs = helper.reset_cutoffs(cutoffs,df,min_samples_leaf)
        return cutoffs
    
        
    def train(self, train_df, max_predicates_per_rule=5, gamma=0.7, theta=0.1, ub=0.98, min_conf=1,
              max_perdicts4rule_mining = 75, max_times4rule_mining = 5,
              set_seed=False, use_multicore=False, verbose=True):
        """
        learn invariant rules from data
        
        Parameters
        ----------
        train_df : DataFrame
            the training data
        max_predicates_per_rule : int
            max number of predicates in a generated rule
        gamma : float in [0,1]
            minimum fraction multiplier for predicate in rule generation, less rules will be generated with a larger value  
        theta : float in [0,1]
            lower bound for the support of a predicate 
        ub : float in [0,1]
            upper bound for the support of a predicate, used only for efficiency purpose
        min_conf : float in [0,1]
            minimum confidence for a rule
        max_perdicts4rule_mining : int
            the max number of predicts can be allowed in a rule mining process (in order to speed up mining process)
        max_times4rule_mining : int
            the max number for the rule mining process, only use for when number of generated predicates is larger than max_perdicts_for_rule_mining
        set_seed : Bool
            whether set random seed for reproducibility
        use_multicore : Bool
            whether use multiple cores for parallel computing, if set to True, the number of cores to use equals to max_times4rule_mining
        verbose : Bool
            whether print progress information during training
        
        Returns
        -----------------
        AssociativePredicateRules
            self
        
        """
        if set_seed:
            np.random.seed(123)
            random.seed(1234)
        
        self._min_conf = min_conf
        self._max_predicates_per_rule = max_predicates_per_rule
        
        min_samples_predicate = int(theta * len(train_df))
        df = train_df.copy()

        cutoffs = self._propose_cutoff_values(df, min_samples_leaf=min_samples_predicate)
#             
        for feat in self._cont_feats:
            if feat not in cutoffs:
                df.drop(columns=feat,inplace=True)
                continue
            
            vals2preserve = [val_pri_pair[0] for val_pri_pair in cutoffs[feat]]
            vals2preserve.sort()
            # print(feat,vals2preserve)           
            for j in range(len(vals2preserve)):
                if j == 0:
                    new_feat = feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] < vals2preserve[j], new_feat] = 1
                if j > 0:
                    new_feat = str(vals2preserve[j-1]) + '<=' + feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[(df[feat]<vals2preserve[j]) & (df[feat]>=vals2preserve[j-1]), new_feat] = 1
                if j == len(vals2preserve)-1:
                    new_feat = feat + '>=' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] >= vals2preserve[j], new_feat] = 1
            df.drop(columns=feat,inplace=True)

            
        low_feats = []
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])
            if support < min_samples_predicate and entry.find('<') == -1 and entry.find('>') == -1:
                low_feats.append( entry )
        # print('low_feats',low_feats)
        
        start_index = 0
        combine_list = []
        for i in range(1,len(low_feats)):
            df.loc[:,'tempt'] = 0
            for j in range(start_index,i):
                df.loc[df[low_feats[j]]==1, 'tempt'] = 1
            left_support = len(df.loc[df['tempt']==1,:])
            # print(low_feats[i],left_support,min_samples_predicate)
             
            if left_support >= min_samples_predicate:
                df.loc[:,'tempt'] = 0
                for j in range(i+1,len(low_feats)):
                    df.loc[df[low_feats[j]]==1, 'tempt'] = 1
                right_support = len(df.loc[df['tempt']==1,:])
                if right_support >= min_samples_predicate:
                    combine_list.append(low_feats[start_index:i+1])
                    start_index = i+1
                else:
                    combine_list.append(low_feats[start_index:])
                    break
        df = df.drop(columns='tempt',errors='ignore')
        # print('combine list',combine_list)
        
        for entry2combine in combine_list:
            new_feat = entry2combine[0]
            for i in range(1,len(entry2combine)):
                new_feat += ' or ' + entry2combine[i]
            df[new_feat] = 0
            for feat in entry2combine:
                df.loc[df[feat]==1, new_feat] = 1
                df = df.drop(columns=feat)
        
        # print('df shape',df.shape)
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])/len(df)
            if support > ub or support < theta:
                # if verbose:
                #     print('drop',entry, len( df.loc[df[entry]==1,:])/len(df) )
                df = df.drop(columns=entry)
            # else:
            #     if verbose:
            #         print('keep',entry, len( df.loc[df[entry]==1,:])/len(df) )
        
        if verbose:
            print('number of generated predicates:',df.shape[1])
        
        ## start invariant rule mining
        rules, self._item_dict = RuleMiner.mine_rules(df, max_len=max_predicates_per_rule, gamma=gamma,
                                                            theta=theta,max_perdicts_for_rule_mining = max_perdicts4rule_mining,
                                                            min_conf = min_conf,
                                                            max_times_for_rule_mining = max_times4rule_mining,
                                                            use_multicore=use_multicore,verbose=verbose)
        self._rules = sorted(rules, key=lambda t: t.size())
        self._rule_linkdict = helper.link_rules(self._rules, max_predicates_per_rule)
        
        if verbose:
            print('number of generated rules',len(self._rules)+len(self._signals))
        return self
    
    def _update_duplicate_rules(self, rule, dupset):
        rules = self._rule_linkdict.get(str(rule),[])
        dupset.update(set(rules))
        for lr in rules:
            self._update_duplicate_rules(str(lr), dupset)
             
    
    @override
    def score_samples(self,df,use_boundary_rules=False, use_cores=1):
        """
        Same to the predict method
        """
        return self.predict(df,use_boundary_rules, use_cores)
        
    @override
    def predict(self,df,use_boundary_rules=False, use_cores=1):
        """
        Predict the anomaly scores for data
        
        Parameters
        ----------
        df : DataFrame
            the test data
        use_boundary_rules : bool
            whether use boundary rules
        kappa : float in (0,inf)
            the scaling factor for the contribution of confidence while computing the anomaly score for a violated rule
        use_cores : int, default is 1
            number of cores to use
            
        Returns
        -------
        ndarray 
            the anomaly scores for each row in df
        """
        test_df = df.copy()
        sigstd = {}
        for signal in self._cont_signals:
            sigstd[signal.name] = signal.std_value
        test_df = helper.parse_predicates(test_df,self._item_dict,sigstd)
        
        test_df.loc[:,'anomaly_score'] = 0
        if use_boundary_rules:
            for signal in self._signals:
                scores = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(signal,sigstd,))
                test_df.loc[:,'anomaly_score'] += scores
                
        if use_cores <= 1:
            rule2ignore = {}
            for rule in self._rules:
                rule2ignore[str(rule)] = set()
                
            for i in range(len(self._rules)):
                rule = self._rules[i]
                ignorelist = rule2ignore[str(rule)]
                
                test_df.loc[:,'antecedent'] = 1
                test_df.loc[:,'consequent'] = 0
                
                for item in rule.antec:
                    test_df.loc[test_df[ self._item_dict[item] ]!=1,  'antecedent'] = 0
    
                for item in rule.conseq:
                    test_df.loc[:,  'consequent'] += 1-test_df[ self._item_dict[item] ].values
                    
                scores4rule = np.multiply(test_df['antecedent'].values,test_df['consequent'].values)*rule.support
                scores4rule[list(ignorelist)] = 0
                test_df.loc[:,'anomaly_score'] += scores4rule
                
                indices = np.where(scores4rule>0)[0]
                # print('indices',indices)
                dups = set()
                self._update_duplicate_rules(rule, dups)
                for dup in dups:
                    if str(dup) not in rule2ignore.keys():
                        print(dup)
                    else: 
                        rule2ignore[str(dup)].update(set(indices))
            return test_df.loc[:,'anomaly_score'].values
        else:
            div_num = len(self._rules)//use_cores
            manager = multiprocessing.Manager()
            results = manager.list()
            jobs = []
            
            rule_clusters = [ [] for i in range(use_cores) ]
            
            rule_dict = {}
            for rule in self._rules:
                rule_dict[str(rule)] = rule
            
            dup_rules = set()
            split_index = 0
            for rule in self._rules:
                if str(rule) in dup_rules:
                    continue
                rule_clusters[split_index].append(rule)
                dups = set()
                self._update_duplicate_rules(rule, dups)
                for dup in dups:
                    if dup not in dup_rules:
                        rule_clusters[split_index].append(rule_dict[dup])
                dup_rules.update(dups)
                
                if len(rule_clusters[split_index]) >= div_num and split_index < use_cores-1:
                    split_index += 1
                    
            for i in range(use_cores):
                rules = rule_clusters[i]
                p = multiprocessing.Process(target=self._parallel_rule_checking, args=(test_df,rules,results))
                jobs.append(p)
                p.start()
        
            for proc in jobs:
                proc.join()
            
            results = np.array(results)
            results = np.sum(results,axis=0)
            return np.add(results,test_df['anomaly_score'].values)
        
    def _parallel_rule_checking(self,df,rules,results):
        test_df = df.copy()
        test_df['tempt'] = 0
        rule2ignore = {}
        for rule in rules:
            rule2ignore[str(rule)] = set()
            
        for i in range(len(rules)):
            rule = rules[i]
            
            ignorelist = rule2ignore[str(rule)]
            test_df.loc[:,'antecedent'] = 1
            test_df.loc[:,'consequent'] = 0
            
            for item in rule.antec:
                test_df.loc[test_df[ self._item_dict[item] ]!=1,  'antecedent'] = 0

            for item in rule.conseq:
                test_df.loc[:,  'consequent'] += 1-test_df[ self._item_dict[item] ].values
                
            scores4rule = np.multiply(test_df['antecedent'].values,test_df['consequent'].values)*rule.support
            scores4rule[list(ignorelist)] = 0
            test_df.loc[:,'tempt'] += scores4rule
            
            indices = np.where(scores4rule>0)[0]
            # print('indices',indices)
            dups = set()
            self._update_duplicate_rules(rule, dups)
            for dup in dups:
                if str(dup) not in rule2ignore.keys():
                    rule2ignore[str(dup)] = set(indices)
                else: 
                    rule2ignore[str(dup)].update(set(indices))
        results.append(test_df.loc[:,'tempt'].values)
    
    def get_anomaly_explanation(self,df,use_boundary_rules=False):
        """
        get the explanation for anomalies
        
        Parameters
        ----------
        df : DataFrame
            the anomalous data
        use_boundary_rules : bool
            whether use boundary rules
            
        Returns
        -------
        AnomalyExplanation 
            anomaly explanation
        """
        test_df = df.copy()
        sigstd = {}
        for signal in self._cont_signals:
            sigstd[signal.name] = signal.std_value
        test_df = helper.parse_predicates(test_df,self._item_dict,sigstd)
        
        exp = AnomalyExplanation()
        
        if use_boundary_rules:
            for signal in self._signals:
                scores = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(signal,sigstd,)).to_numpy()
                indices = np.where(scores>0)[0]
                # print(scores)
                # print(indices)
                for i in indices:
                    exp.add_record(signal.name, i, scores[i], helper.boundary_rule(signal),[signal.name])
        
        
        for i in test_df.index:
            # print(i,'/',len(test_df))            
            dups = set()
            for rule in self._rules:
                if str(rule) in dups:
                    continue
                antec_satisfy = True
                conseq_satisfy = True
                
                for item in rule.antec:
                    if test_df.loc[i,self._item_dict[item]] != 1:
                        antec_satisfy = False
                        break
                
                if antec_satisfy:
                    for item in rule.conseq:
                        if test_df.loc[i,self._item_dict[item]] != 1:
                            conseq_satisfy = False
                            feat = helper.extract_feat_from_predicate(self._item_dict[item])
                            score = rule.support*(1-test_df.loc[i,self._item_dict[item]])
                            if feat is not None:
                                exp.add_record(feat, i, score, str(rule),rule.extract_feats())
                            
                    if conseq_satisfy == False:
                        # print('before update', dups)
                        self._update_duplicate_rules(rule, dups)
                        # print('after update', dups)
        return exp
              
    def export_rules(self, filepath):
        """
        export rules to file
        
        Parameters
        ----------
        filepath : string
            the file path
        """
        with open(filepath,'w') as myfile:
            for signal in self._signals:
                myfile.write(helper.boundary_rule(signal) + '\n')
            for rule in self._rules:
                myfile.write(str(rule) + '\n')
            myfile.close()
    
    
    def get_num_rules(self):
        """
        get number of rules
        
        Returns
        -------
        int
            the number of rules
        """
        return len(self._rules)+len(self._signals)
    
    def score(self):
        pass
    
    @override
    def save_model(self,model_path=None, model_id=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
        """
        
        model_path = super().save_model(model_path, model_id)
        pickle.dump(self._rules,open(model_path+'/rules.pkl','wb'))
        pickle.dump(self._item_dict,open(model_path+'/item_dict.pkl','wb'))
        config_dict = {'max_predicates_per_rule':self._max_predicates_per_rule}
        pickle.dump(config_dict,open(model_path+'/config_dict.pkl','wb'))
    
    @override
    def load_model(self,model_path=None, model_id=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
            
        Returns
        -------
        InvariantRuleModel
            self
        """
        model_path = super().load_model(model_path, model_id)
        self._item_dict = pickle.load(open(model_path+'/item_dict.pkl','rb'))
        
        rules = pickle.load(open(model_path+'/rules.pkl','rb'))
        self._rules = sorted(rules, key=lambda t: t.size())
        
        config_dict = pickle.load(open(model_path+'/config_dict.pkl','rb'))
        self._max_predicates_per_rule = config_dict['max_predicates_per_rule']
        self._rule_linkdict = helper.link_rules(self._rules, self._max_predicates_per_rule)
        return self

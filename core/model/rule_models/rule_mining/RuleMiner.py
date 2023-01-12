from . import MISTree
import pandas as pd
from . import RuleGenerator
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import multiprocessing
from ..rule import Rule
# import time

def _mining(data,gamma, max_k, theta,min_conf,index_dict):
    minSup_dict = {}
    min_num = len(data)*theta
    
    for entry in data:
        minSup_dict[ index_dict[entry]  ] = max( gamma*len(data[data[entry] == 1]), min_num )
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
            
    item_count_dict = MISTree.count_items(dataset)
    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)
     
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)
    L = RuleGenerator.arrangePatterns(freq_patterns, support_data, item_count_dict, max_k, MIN)
    rules = RuleGenerator.generateRules(len(data), L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=min_conf)
    return rules,freq_patterns,support_data

def _parallel_mining(data,gamma, max_k, theta,min_conf,index_dict,frequent_patterns_all,support_data_all,rules_all):
    minSup_dict = {}
    min_num = len(data)*theta
    for entry in data:
        minSup_dict[ index_dict[entry]  ] = max( gamma*len(data[data[entry] == 1]), min_num )
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
            
    item_count_dict = MISTree.count_items(dataset)
    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)
     
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)
    L = RuleGenerator.arrangePatterns(freq_patterns, support_data, item_count_dict, max_k, MIN)    
    rules = RuleGenerator.generateRules(len(data), L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=min_conf)
    frequent_patterns_all.extend(freq_patterns)
    support_data_all.update(support_data)
    rules_all.extend(rules)

def _multi_mine(df,gamma, max_k, theta,min_conf,index_dict,max_perdicts_for_rule_mining, max_times_for_rule_mining,use_multicore,verbose):
    if use_multicore:
        manager = multiprocessing.Manager()
        frequent_patterns_all = manager.list()
        support_data_all = manager.dict()
        rules_all = manager.list()
        jobs = []
        for i in range(max_times_for_rule_mining):
            if verbose:
                print('Start rule ming process:',i+1,"/",max_times_for_rule_mining)
            data = df.sample(n=max_perdicts_for_rule_mining,axis='columns')
            p = multiprocessing.Process(target=_parallel_mining, args=(data,gamma, max_k, theta,min_conf,index_dict,
                                                                       frequent_patterns_all,support_data_all,rules_all))
            jobs.append(p)
            p.start()
        
        for proc in jobs:
            proc.join()
    else:
        frequent_patterns_all = []
        support_data_all = {}
        rules_all = []
        for i in range(max_times_for_rule_mining):
            if verbose:
                print('Start rule ming process:',i+1,"/",max_times_for_rule_mining)
            data = df.sample(n=max_perdicts_for_rule_mining,axis='columns')
            rules,freq_patterns,support_data = _mining(data,gamma, max_k, theta,min_conf,index_dict)
            frequent_patterns_all.extend(freq_patterns)
            support_data_all.update(support_data)
            rules_all.extend(rules)
    
    ##filter duplicated rules    
    final_rules = []
    rule_dict = {}
    for rule in rules_all:
        if rule.pset() not in rule_dict.keys():
            final_rules.append(rule)
            rule_dict[rule.pset()] = [( set(rule.antec),set(rule.conseq) )]
        else:
            exist_rules = rule_dict[rule.pset()]
            for ex_rule in exist_rules:
                ant,con = ex_rule
                dup = False
                if set(ant) == set(rule.antec) or set(con) == set(rule.conseq):
                    dup = True
                    break
            if not dup:
                final_rules.append(rule)
                rule_dict[rule.pset()].append( ( set(rule.antec),set(rule.conseq) ) )
    return final_rules

def _mining_fp(data, max_len, theta,min_conf,index_dict):
    for entry in data:
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
            
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = fpgrowth(df, min_support=theta,use_colnames=True,max_len=int(max_len))
      
    df = association_rules(frequent,metric='confidence',min_threshold=min_conf)
    rules = []
    for i in range(len(df)):
        antecedents = df.loc[i,'antecedents']
        consequents = df.loc[i,'consequents']
        confidence = df.loc[i,'confidence']
        support = df.loc[i,'support']
        rule = Rule(antecedents,consequents,confidence,support)
        rules.append(rule)
    return rules

def mine_rules(df, max_len, gamma, theta, min_conf, max_perdicts_for_rule_mining, max_times_for_rule_mining,use_multicore,verbose):
    index_dict = {}
    item_dict = {}
    index = 100
    for entry in df:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    if gamma == 0:
        data = df.copy()
        rules = _mining_fp(data, max_len, theta,min_conf,index_dict)
    else:
        if df.shape[1] <= max_perdicts_for_rule_mining or max_times_for_rule_mining<=1:
            data = df.copy()
            rules,_,_ = _mining(data, gamma, max_len-1, theta,min_conf,index_dict)
        else:
            rules = _multi_mine(df,gamma, max_len-1, theta,min_conf,index_dict,max_perdicts_for_rule_mining, max_times_for_rule_mining,use_multicore,verbose)
    
    for rule in rules:
        rule.set_itemdict(item_dict)
    return rules, item_dict
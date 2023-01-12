import numpy as np
from ...preprocessing.signals import ContinuousSignal

def search_insert_position(vals, val2insert):
    pos = None
    for i in range(len(vals)):
        if val2insert <= vals[i]:
            pos = i
    
    if pos is None:
        pos = len(vals)
    return pos

def reset_cutoffs(cutoffs,df,min_samples):
    preserved_cutoffs = {}
    for feat in cutoffs.keys():
        if len(cutoffs[feat]) == 0:
            preserved_cutoffs[feat] = []
            continue
        
        value_priority_pairs = cutoffs[feat]
        sorted_value_priority_pairs = sorted(
            value_priority_pairs,
            key=lambda t: [t[1],-t[0]],
            reverse=True
        )
        # print(feat,sorted_value_priority_pairs)
        for i in range(len(sorted_value_priority_pairs)):
            init_val = sorted_value_priority_pairs[i][0]
            if len( df.loc[ df[feat]<init_val,:] ) >= min_samples and len( df.loc[ df[feat]>init_val,:] ) >= min_samples:
                sorted_value_priority_pairs = sorted_value_priority_pairs[i:]
                break
        
        
        vals = [pair[0] for pair in sorted_value_priority_pairs]
        pair2preserve = [sorted_value_priority_pairs[0]]
        val2preserve = []
        val2preserve.append(vals[0])
        for i in range(1, len(vals)):
            pos = search_insert_position(val2preserve, vals[i])
            if pos == 0:
                if len( df.loc[ (df[feat]>=vals[i]) & (df[feat]<val2preserve[0]),:] ) >= min_samples and \
                    len( df.loc[ df[feat]<vals[i],:] ) >= min_samples:
                    val2preserve.insert(0,vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
            elif pos == len(val2preserve):
                if len( df.loc[ (df[feat]<vals[i]) & (df[feat]>=val2preserve[-1]),:] ) >= min_samples and \
                    len( df.loc[ df[feat]>=vals[i],:] ) >= min_samples:
                    val2preserve.append(vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
            else:
                if len( df.loc[ (df[feat]<vals[i]) & (df[feat]>=val2preserve[pos-1]),:] ) >= min_samples and \
                    len(df.loc[(df[feat] < val2preserve[pos]) & (df[feat] >= vals[i]),:]) >= min_samples:
                    val2preserve.insert(pos, vals[i])
                    pair2preserve.append(sorted_value_priority_pairs[i])
        preserved_cutoffs[feat] = pair2preserve
    return preserved_cutoffs


def extract_cutoffs(dtree,feats):
    n_nodes = dtree.node_count
    children_left = dtree.children_left
    children_right = dtree.children_right
    feature = dtree.feature
    threshold = dtree.threshold
    node_samples = dtree.n_node_samples
    impurity = dtree.impurity
    N = dtree.n_node_samples[0]
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    cutoffs = []
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            
    for i in range(n_nodes):
        if is_leaves[i] == False:
            c_l, c_r = children_left[i], children_right[i]
            N_t, N_t_L, N_t_R = node_samples[i], node_samples[c_l], node_samples[c_r]
            imp, imp_l, imp_r = impurity[i], impurity[c_l], impurity[c_r]
            impurity_decrease = N_t / N * (imp - N_t_R/N_t*imp_r - N_t_L/N_t*imp_l)

            if type(feats) == list:
                # print(feature[i])
                # print(feat)
                cutoffs.append( (feats[feature[i]],threshold[i], impurity_decrease) )
            else:
                cutoffs.append( (feats,threshold[i], impurity_decrease) )
    return cutoffs


def link_rules(rules,max_predicates_per_rule):
    L = []
    for _ in range(max_predicates_per_rule-1):
        L.append([])

    for rule in rules:
        L[rule.size()-2].append( rule )
    
    link_dict = {}
    for i in range(0, len(L)-2):
        LPrev = L[i]
        LNext = L[i+1]
        for shortRule in LPrev:
            for largeRule in LNext:
                if set(shortRule.antec) == set(largeRule.antec) and set(shortRule.conseq).issubset( set(largeRule.conseq) ):
                    linked_list = link_dict.get(str(shortRule),[])
                    linked_list.append(str(largeRule))
                    link_dict[str(shortRule)] = linked_list
    return link_dict



def fuzzy_membership(x,predicate,sigstd):
    score = 0
    if predicate.find(' or ') != -1:
        subitems = predicate.split(' or ')
        satisfy = False
        for subitem in subitems:
            if x[subitem]==1:
                satisfy = True
                break
        if satisfy:
            score = 1
        else:
            score = 0
    else:
        if predicate.find('<=') != -1:
            pos1, pos2 = predicate.find('<='), predicate.rfind('<')
            lb = float(predicate[:pos1])
            ub = float(predicate[pos2+1:])
            feat = predicate[pos1+2:pos2]
            if x[feat] >= lb and x[feat] <= ub:
                score = 1
            elif x[feat] < lb:
                score = 1-max(1, (lb-x[feat])/(sigstd[feat]+1e-5) )
            elif x[feat] > ub:
                score = 1-max(1, (x[feat]-ub)/(sigstd[feat]+1e-5) )
        elif predicate.find('<') != -1:
            pos = predicate.find('<')
            ub = float(predicate[pos+1:])
            feat = predicate[:pos]
            if x[feat] <= ub:
                score = 1
            else:
                score = 1-max(1, (x[feat]-ub)/(sigstd[feat]+1e-5) )
        elif predicate.find('>=') != -1:
            pos = predicate.find('>=')
            lb =  float(predicate[pos+2:])
            feat = predicate[:pos]
            if x[feat] >= lb:
                score = 1
            else:
                score = 1-max(1, (lb-x[feat])/(sigstd[feat]+1e-5) )
        else:
            if x[predicate] == 1:
                score = 1
            else:
                score = 0        
    return score
        

def parse_predicates(df, item_dict, sigstd):
    for item in item_dict.values():
        scores = df.apply(fuzzy_membership,axis=1,args=(item,sigstd,))
        df[item] = scores
    return df

def boundary_anomaly_scoring(x,signal,sigstd):
    if isinstance(signal, ContinuousSignal):
        lb = min(signal.min_value, signal.mean_value-3*signal.std_value)
        ub = max(signal.max_value, signal.mean_value+3*signal.std_value)
        feat = signal.name
        # print(feat,lb,ub)
        if x[feat] >= lb and x[feat] <= ub:
            score = 0
        elif x[feat] < lb:
            score = max(1, (lb-x[feat])/(sigstd[feat]+1e-8) )
        elif x[feat] > ub:
            score = max(1, (x[feat]-ub)/(sigstd[feat]+1e-8) )
    else:
        satisfy = False
        for entry in signal.get_onehot_feature_names():
            if x[entry] == 1:
                satisfy = True
                break
        if satisfy:
            score = 0
        else:
            score = 1
    return score


def extract_feat_from_predicate(item):
    if item.find(' or ') != -1:
        return None
    else:
        if item.find('<=') != -1:
            pos1, pos2 = item.find('<='), item.rfind('<')
            feat = item[pos1+2:pos2]
        elif item.find('<') != -1:
            pos = item.find('<')
            feat = item[:pos]
        elif item.find('>=') != -1:
            pos = item.find('>=')
            feat = item[:pos]
        else:
            pos = item.rfind('=')
            feat = item[:pos]
    return feat



def boundary_rule(signal):
    if isinstance(signal, ContinuousSignal):
        lb = min(signal.min_value, signal.mean_value-3*signal.std_value)
        ub = max(signal.max_value, signal.mean_value+3*signal.std_value)
        strOut = str(lb)+'<='+signal.name+'<='+str(ub)            
    else:
        strOut = signal.name + ' in ' + str(signal.get_onehot_feature_names())
    # strOut += ', sup 1, conf 1'
    return strOut  
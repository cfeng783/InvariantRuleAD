from ..rule import Rule

def arrangePatterns(freq_patterns, support_data, item_count_dict, max_k, MIN):
    """arrange frequent patterns
    """
    L = []
    for _ in range(max_k+1):
        L.append([])
    
    for item in item_count_dict:
        if item_count_dict[item] >= MIN:
            key =frozenset([item])
            support_data[key] = item_count_dict[item]
            L[0].append(key)
    
    for pattern in freq_patterns:
        L[len(pattern)-1].append( frozenset(pattern) )
    return L 

def generateRules(MAX, L, support_data, MIN_freq_item_header_table,min_sup, min_confidence):
    """Create the association rules
    L: list of frequent item sets
    support_data: support data for those itemsets
    min_confidence: minimum confidence threshold
    """
    rules = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
#             print "freqSet", freqSet, 'H1', H1
            if (i > 1):
                rules_from_conseq(MAX,freqSet, H1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
            else:
                calc_confidence(MAX,freqSet, H1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
    return rules


def calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup):
    item_mis_tuples = []
    for item in itemlist:
        im_tuple = (item, min_sup[item])
        item_mis_tuples.append(im_tuple)
        
    item_mis_tuples.sort(key=lambda tup: (tup[1],tup[0]))
    
    count = 0
    entry = MIN_freq_item_header_table[  item_mis_tuples[0][0] ]
    node = entry.node_link
    while node != None:
        i=1
        parent = node.parent_link
        while parent.parent_link != None and i<len(item_mis_tuples):
            if parent.item == item_mis_tuples[i][0]:
                i+=1
            parent = parent.parent_link
        
#         print 'i:' + str(i) + ' len:' + str( len(item_mis_tuples) )
        
        if i == len(item_mis_tuples):
            count += node.count
        
        node = node.node_link 
                     
    return count 

def calc_confidence(MAX, freqSet, H, support_data, rules, MIN_freq_item_header_table,min_sup, min_confidence):
    "Evaluate the rule generated"
    pruned_H = []
    for conseq in H:
        if freqSet-conseq not in support_data:
            itemlist = list(freqSet - conseq)
            support_data[freqSet - conseq] = calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup)
#         
#         conf = support_data[freqSet] / support_data[freqSet - conseq]
#         if conf >= min_confidence:
#             rules.append((freqSet - conseq, conseq, conf))
#             pruned_H.append(conseq)
#         itemlist = list(freqSet)
#         count = calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup)
#         if count != support_data[freqSet]:
#             print itemlist
#             print 'count: ' + str(count) + ' support: ' + str(support_data[freqSet])
#             
#         itemlist = list(freqSet-conseq)
#         count = calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup)
#         if count != support_data[freqSet-conseq]:
#             print itemlist
#             print 'count: ' + str(count) + ' support: ' + str(support_data[freqSet-conseq])
          
#         if freqSet-conseq in support_data:
        conf = support_data[freqSet]*1.0 / support_data[freqSet - conseq]
        support = support_data[freqSet]/MAX
        if conf >= min_confidence:
#             print conf
            rules.append( Rule(freqSet - conseq, conseq, conf, support) )
            pruned_H.append(conseq)
    return pruned_H

def aprioriGen(freq_sets, k):
    "Generate the joint transactions from candidate sets"
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


def rules_from_conseq(MAX, freqSet, H, support_data, rules, MIN_freq_item_header_table,min_sup, min_confidence):
    "Generate a set of candidate rules"
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calc_confidence(MAX, freqSet, Hmp1,  support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
        if len(Hmp1) > 1:
            rules_from_conseq(MAX, freqSet, Hmp1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
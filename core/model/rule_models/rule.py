# import math
from . import helper

class Rule(object):
    '''
    An Associative Predicate Rule
    
    Parameters
    ----------
    antec : list
        list of predicates in the antecedent set
    conseq : list
        list of predicates in the consequent set
    conf : float in [0,1]
        the confidence of the rule
    support : float in [0,1]
        the support of the rule
    '''
    def __init__(self, antec, conseq, conf, support):
        '''
        Constructor
        '''
        self._antec = antec
        self._conseq = conseq
        self._conf = conf
        self._support = support
        self._item_dict = None
    
    def __str__(self):
        strRule = ''
        for item in self._antec:
            strRule += self._item_dict[item] + ' and '
        strRule = strRule[0:len(strRule)-5] 
        strRule += ' ---> '
        for item in self._conseq:
            strRule += self._item_dict[item] + ' and '
        strRule = strRule[0:len(strRule)-5]
        # strRule += f', sup {rule[3]:.2f}, conf {rule[2]:.2f}'
        return strRule
    
    @property
    def antec(self):
        return self._antec
    
    @property
    def conseq(self):
        return self._conseq
    
    @property
    def conf(self):
        return self._conf
    
    @property
    def support(self):
        return self._support
    
    def size(self):
        """
        get the number of predicates in the rule 
            
        Returns
        -------
        int 
            the number of predicates
        """
        return len(self._antec) + len(self._conseq)
    
    def pset(self):
        """
        get the set of predicates in the rule 
            
        Returns
        -------
        frozenset 
            the set of predicates
        """
        return frozenset( set(self._antec) | set(self._conseq) )
  
    def set_itemdict(self,item_dict):
        self._item_dict = {}
        for item in self.pset():
            self._item_dict[item] = item_dict[item]
    
    def extract_feats(self):
        antec_feats = []
        for item in self._antec:
            antec_feats.append( helper.extract_feat_from_predicate(self._item_dict[item]) )
        conseq_feats = []
        for item in self._conseq:
            conseq_feats.append( helper.extract_feat_from_predicate(self._item_dict[item]) )
        return antec_feats+conseq_feats
                
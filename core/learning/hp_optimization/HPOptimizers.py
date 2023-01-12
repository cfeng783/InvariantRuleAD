import itertools,collections
from .Hyperparameter import HyperparameterType


class RandomizedGridSearch(object):
    '''
    The utility class for hyperparameters tuning of ML models based on Randomized Grid Search.
    
    Parameters
    ----------
    model : BaseModel
        The model, should be an object extends BaseModel
    hyperparameters : list
        The list of hyperparameters of the ML model
    train_data : ndarray or list of ndarray
        the training data
    val_data : ndarray or list of ndarray
        the validation data

    '''
    
    def __init__(self, model, hyperparameters, train_data, val_data):
        '''
        Constructor
        '''
        self._model = model
        self._hyperparameters = hyperparameters
        
        self._train_data = train_data
        self._val_data = val_data
    
    def _eval(self, **args): 
        self._model.train(self._train_data,self._val_data,**args)
        score = self._model.score(self._val_data)
        if score > self._best_score:
            self._model.save_model()
            self._best_score = score
            self._best_config = args.copy()
        return score
        
    
    def run(self, n_searches ,verbose=0):
        '''
        Run the randomized grid search algorithm to find the best hyperparameter configuration for the ML model
        
        Parameters
        ----------
        n_searches : int
            the number of searches
        verbose : int, default is 0
            higher level of verbose prints more messages during running the algorithm
        
        Returns
        -------
        BaseModel 
            the optimized model
        dict
            the hyperparameter configuration of the optimized model
        float
            the best score achieved by the optimized model
        
        '''
        self._best_score = float('-inf')
        
        random_mode = False
        for hp in self._hyperparameters:
            if hp.hp_type != HyperparameterType.Const and hp.hp_type != HyperparameterType.Categorical:
                random_mode = True
                break
        if random_mode == False:
            param_lists = []
            for hp in self._hyperparameters:
                param_lists.append(hp.getAllValues())
            candidates = list(itertools.product(*param_lists))
            if len(candidates) > n_searches:
                random_mode = True
       
        if random_mode:
            candidates = []
            for _ in range(n_searches):
                param_list = []
                for hp in self._hyperparameters:
                    param_list.append(hp.getValue())
                isDup = False
                for can in candidates:
                    if collections.Counter(can) == collections.Counter(param_list):
                        isDup = True
                        break
                if isDup == False:
                    candidates.append(param_list)
        
        for can in candidates:
            i = 0
            can_dict = {}
            for hp in self._hyperparameters:
                can_dict[hp.name] = can[i]
                i += 1
            score = self._eval(**can_dict)
            if verbose > 0:
                print('score:', score, can_dict)
        self._model.load_model()
        return self._model, self._best_config, self._best_score

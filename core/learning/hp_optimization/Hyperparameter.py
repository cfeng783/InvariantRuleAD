from abc import ABC,abstractmethod
import random
from enum import Enum

class HyperparameterType(Enum):
    UniformInteger = 301
    UniformFloat = 302
    Categorical = 303
    Const = 304

class baseHyperparameter(ABC):
    '''
    The base class for Hyperparameters
    
    Parameters
    ----------
    name : string
        the name of the hyperparameter 
    hp_type : HyperparameterType
        the type of the hyperparameter
    '''

    def __init__(self,name,hp_type):
        '''
        Constructor
        '''
        self._name = name
        self._hp_type = hp_type
    
    @abstractmethod
    def getValue(self):
        '''
        Get a value of the hyperparameter
        
        Returns
        -------
        Object 
            the value
        '''
        pass
    
    @abstractmethod
    def getAllValues(self):
        '''
        Get all possible values for the hyperparameter
        
        Returns
        -------
        list or tuple
            the all possible values
        '''
        pass
    
    @property
    def name(self):
        """Get the name."""
        return self._name
    
    @property
    def hp_type(self):
        """Get the type."""
        return self._hp_type

class UniformIntegerHyperparameter(baseHyperparameter):
    '''
    The uniform integer hyperparameter class
    
    Parameters
    ----------
    name : string
        the name of the hyperparameter 
    lb : int
        the lower bound of the hyperparameter, inclusive
    ub : int
        the upper bound of the hyperparameter, inclusive
    '''

    def __init__(self, name, lb, ub):
        '''
        Constructor
        '''
        self.bot = lb
        self.top = ub
        super().__init__(name,HyperparameterType.UniformInteger)
    
    def getValue(self):
        '''
        Get a random value of the hyperparameter
        
        Returns
        -------
        int 
            a random value between [lb,ub]
        '''
        return random.randint(self.bot,self.top)
    
    def getAllValues(self):
        '''
        Get all possible values for the hyperparameter
        
        Returns
        -------
        tuple 
            a tuple including the lower bound and upper bound of the hyperparameter
        '''
        return (self.bot,self.top)

class UniformFloatHyperparameter(baseHyperparameter):
    '''
    The uniform float hyperparameter class
    
    Parameters
    ----------
    name : string
        the name of the hyperparameter 
    lb : float
        the lower bound of the hyperparameter, inclusive
    ub : float
        the upper bound of the hyperparameter, inclusive
    '''


    def __init__(self, name, lb, ub):
        '''
        Constructor
        '''
        self.bot = lb
        self.top = ub
        super().__init__(name,HyperparameterType.UniformFloat)
    
    def getValue(self):
        '''
        Get a random value of the hyperparameter
        
        Returns
        -------
        float 
            a random value between [lb,ub]
        '''
        return random.uniform(self.bot,self.top)
    
    def getAllValues(self):
        '''
        Get all possible values for the hyperparameter
        
        Returns
        -------
        tuple 
            a tuple including the lower bound and upper bound of the hyperparameter
        '''
        return (self.bot,self.top)
        

class CategoricalHyperparameter(baseHyperparameter):
    '''
    The categorical hyperparameter class
    
    Parameters
    ----------
    name : string
        the name of the hyperparameter 
    value_list : list
        the list of all possible values of the hyperparameter
    '''
    
    def __init__(self, name, value_list):
        '''
        Constructor
        '''
        self.value_list = value_list
        super().__init__(name,HyperparameterType.Categorical)
    
    def getValue(self):
        '''
        Get a random value of the hyperparameter
        
        Returns
        -------
        object 
            a random value in value_list
        '''
        
        idx = random.randint(0,len(self.value_list)-1)
        return self.value_list[idx]
    
    def getAllValues(self):
        '''
        Get all possible values for the hyperparameter
        
        Returns
        -------
        list 
            the value list
        '''
        return self.value_list


class ConstHyperparameter(baseHyperparameter):
    '''
    The constant hyperparameter class
    
    Parameters
    ----------
    name : string
        the name of the hyperparameter 
    value : object
        the value of the hyperparameter
    '''
    
    def __init__(self, name, value):
        '''
        Constructor
        '''
        self.value = value
        super().__init__(name,HyperparameterType.Const)
    
    def getValue(self):
        '''
        Get the value of the hyperparameter
        
        Returns
        -------
        object 
            the value
        '''
        return self.value
    
    def getAllValues(self):
        '''
        Get all possible values for the hyperparameter
        
        Returns
        -------
        list 
            a list only has one member, the value 
        '''
        return [self.value]  
        
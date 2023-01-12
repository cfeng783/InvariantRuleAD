from abc import ABC, abstractmethod
import tempfile
import os
import warnings

class BaseModel(ABC):
    """
    The base class
    """
    
    @abstractmethod
    def train(self, train_data, val_data=None, **params):
        """
        Create a model based on the give hyperparameters and train the model
        
        Parameters
        ----------
        train_data : ndarray
            the numpy array from where the training samples are extracted
        val_data : ndarray, default is None
            the numpy array from where the validation samples are extracted
        params : dict
            the hyperparameters of the model
        
        Returns
        -------
        BaseModel
            self
        
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Predict outputs for x
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        
        Returns
        -------
        ndarray or list of ndarray
            the output data
        """
        pass
    
    @abstractmethod
    def score(self, data):
        """
        Score the model based on its performance on given data. 
        A higher score indicates a better performance. 
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        
        Returns
        -------
        float
            the score
        """
        pass
    
    # @abstractmethod
    # def get_default_hyperparameters(self):
    #     """
    #     get the default values or ranges of hyperparameters for tuning
    #
    #     Returns
    #     -------
    #     list 
    #         list of Hyperparameters
    #     """
    #     pass
    
    
    def save_model(self, model_path=None, model_id=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
        
        Returns
        -------
        string
            the path to load the model 
        """
        if model_path is None:
            model_path = tempfile.gettempdir()
        else:
            if model_id is None:
                warnings.warn('Please specify model id')
            model_path = model_path+'/'+model_id
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        return model_path
    
    def load_model(self, model_path=None,model_id=None):
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
        string
            the path to load the model 
        """
        if model_path is None:
            model_path = tempfile.gettempdir()
        else:
            if model_id is None:
                warnings.warn('Please specify model id')
            
            model_path = model_path+'/'+model_id
        return model_path
    
    


class AnomalyDetector(ABC):
    """
    The abstract class for all anomaly detectors in predictive_maintenance
    """
    
    @abstractmethod
    def score_samples(self, **params):
        """
        get the anomaly scores for the input samples
        
        Parameters
        ----------
        params : dict
            input parameters
        
        """
        pass
        
    



    
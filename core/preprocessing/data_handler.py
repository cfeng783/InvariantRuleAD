import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
from builtins import isinstance


class TSAEDataHandler():
    '''
    Data Handler for time-series autoencoders
    
    Parameters
    ----------
    sequence_length : int
        the length of sequence
    feats : list of strings
        the target variables
    df_columns : list of strings
        the column names in the DataFrame from where the data is extracted
    '''
    def __init__(self, sequence_length,feats,df_columns):
        # Store the raw data.
        self.df_columns = df_columns
             
        self.column_indices = {name: i for i, name in enumerate(df_columns)}
        self.feats = feats
        self.feat_indices = [self.column_indices[name] for name in feats]
        # Work out the window parameters.
        self.total_window_size = sequence_length
        
        self.feat_index_dict = {name: i for i, name in enumerate(feats)}
        
    def _split_window_for_seq2seq_train(self, features):
        inputs = features
        inputs = tf.stack([inputs[:, :, idx] for idx in self.feat_indices],axis=-1)
        inputs.set_shape([None, self.total_window_size, len(self.feats)])
        outputs = tf.reverse(inputs,[1])
        return inputs, outputs
    
    def _split_window_for_block_train(self, features):
        inputs = features
        inputs = tf.stack([inputs[:, :, idx] for idx in self.feat_indices],axis=-1)
        inputs = tf.reshape(inputs,[-1,self.total_window_size*len(self.feats)])
        return inputs, inputs
    
    def _split_window_for_seq2seq_predict(self, features):
        inputs = features
        
        inputs = tf.stack([inputs[:, :, idx] for idx in self.feat_indices],axis=-1)
        
        inputs.set_shape([None, self.total_window_size, len(self.feats)])
    
        return inputs
    
    
    def _split_window_for_block_predict(self, features):
        inputs = features
        inputs = tf.stack([inputs[:, :, idx] for idx in self.feat_indices],axis=-1)
        inputs = tf.reshape(inputs,[-1,self.total_window_size*len(self.feats)])
        return inputs
    
    def _split_window_for_label(self, features):
        return features
        
    def make_dataset(self, data, mode,  batch_size=256):
        """
        make a tensorflow dataset object for model training and validation
        
        Parameters
        ----------
        data : ndarray or list of ndarray
            the numpy array from where the samples are extracted
        mode : {'seq2seq','block'}
            the mode for time-series
        batch_size : int
            the batch_size
        
        Returns
        -------
        Dataset 
            a tf.data.Dataset object
        """
        if isinstance(data,list):
            cds = None
            for dt in data:
                dt = np.array(dt, dtype=np.float32)
                ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                  data=dt,
                  targets=None,
                  sequence_length=self.total_window_size,
                  sequence_stride=1,
                  shuffle=True,
                  batch_size=batch_size,)
                if mode == 'seq2seq':
                    ds = ds.map(self._split_window_for_seq2seq_train)
                elif mode == 'block':
                    ds = ds.map(self._split_window_for_block_train)
                if cds is None:
                    cds = ds
                else:
                    cds = cds.concatenate(ds)
            return cds
        else:
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                  data=data,
                  targets=None,
                  sequence_length=self.total_window_size,
                  sequence_stride=1,
                  shuffle=True,
                  batch_size=batch_size,)
            if mode == 'seq2seq':
                ds = ds.map(self._split_window_for_seq2seq_train)
            elif mode == 'block':
                ds = ds.map(self._split_window_for_block_train)
            return ds
    
    def extract_data4AD(self, data, mode, stride):
        """
        extract input samples as well as output samples from raw data for anomaly detection
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        mode : {'seq2seq','block'}
            the mode for time-series
        stride : int
            period between successive extracted sequences
        
        Returns
        -------
        ndarray 
            the encoder inputs, matrix of shape = [n_samples, n_input_feats]
        ndarray 
            the output data, shape = [n_samples, n_output_feats]
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=stride,
              shuffle=False,
              batch_size=1,)
        
        if mode == 'seq2seq':
            ds = ds.map(self._split_window_for_seq2seq_train)
        elif mode == 'block':
            ds = ds.map(self._split_window_for_block_train)
        
        x_list, y_list = [], []
        for x, y in ds.as_numpy_iterator():
            x_list.append(x)
            y_list.append(y)
        x, y = np.concatenate(x_list), np.concatenate(y_list)
        return x, y
    
    def extract_data4predict(self, data, mode, stride):
        """
        extract input samples as well as output samples from raw data for ts prediction
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        mode : {'seq2seq','block'}
            the mode for time-series
        stride : int
            period between successive extracted sequences
        
        Returns
        -------
        ndarray 
            the encoder inputs, matrix of shape = [n_samples, n_input_feats]
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=stride,
              shuffle=False,
              batch_size=1,)
        
        if mode == 'seq2seq':
            ds = ds.map(self._split_window_for_seq2seq_predict)
        elif mode == 'block':
            ds = ds.map(self._split_window_for_block_predict)
        
        x_list = []
        for x in ds.as_numpy_iterator():
            x_list.append(x)
        x = np.concatenate(x_list)
        return x
    
     
    def extract_labels(self, data):
        """
        extract labels from raw data
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the labels are extracted
        
        Returns
        -------
        ndarray 
            the labels, matrix of shape = [n_samples, output_width]
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=self.total_window_size,
              shuffle=False,
              batch_size=1,)
        ds = ds.map(self._split_window_for_label)
        
        x_list = []
        for x in ds.as_numpy_iterator():
            x_list.append(x)
        labels = np.concatenate(x_list)
        labels = labels.sum(axis=1)
        labels[labels>0]=1
        return labels
    
    def extractRes(self,obs_x,recon_x, target_vars=None,denormalizer=None):
        """
        extract prediction compared with inputs
        
        Parameters
        ----------
        obs_x : ndarray
            the ground truth data, matrix of shape = [n_samples, n_features]
        recon_x : ndarray
            the reconstructed data, matrix of shape = [n_samples, n_features]
        target_vars : list of string, default is None
            the variables to extract, if None, all feats will be extracted
        denormalizer: DataUtil, default is None
            the DataUtil object to denormalize the data before extraction. If None, no denormalization will be conducted
        
        Returns
        -------
        ndarray 
            the extracted ground truth data, matrix of shape = [n_samples, n_target_vars]
        ndarray 
            the extracted reconstructed data, matrix of shape = [n_samples, n_target_vars]
        """
        
        if target_vars is None:
            target_vars = self.feats
        else:
            for var in target_vars:
                if var not in self.feats:
                    msg = 'Warning, ' + var + ' is not a target feature!'
                    warnings.warn(msg)
                    return None
                
        if denormalizer is not None:
            obs_x = denormalizer.denormalize(obs_x,self.feats)
            recon_x = denormalizer.denormalize(recon_x,self.feats)
        
        target_indices = []
        for target_var in target_vars:
            j = self.feat_index_dict[target_var]
            target_indices.append(j)
            
        return obs_x[:,target_indices], recon_x[:,target_indices]
    
    def plotRes(self,obs_x,recon_x,plot_vars=None,denormalizer=None):
        """
        plot prediction compared with inputs
        
        Parameters
        ----------
        obs_x : ndarray
            the ground truth data, matrix of shape = [n_samples, n_features]
        recon_x : ndarray
            the reconstructed data, matrix of shape = [n_samples, n_features]
        plot_vars : list of string, default is None
            the variables to plot, if None, all feats will be plotted
        denormalizer: DataUtil, default is None
            the DataUtil object to denormalize the data before plot. If None, no denormalization will be conducted
        """
        
        if plot_vars is None:
            plot_vars = self.feats
        else:
            for var in plot_vars:
                if var not in self.feats:
                    msg = 'Warning, ' + var + ' is not a target feature!'
                    warnings.warn(msg)
                    return None
                
        if denormalizer is not None:
            obs_x = denormalizer.denormalize(obs_x,self.feats)
            recon_x = denormalizer.denormalize(recon_x,self.feats)
        
        timeline = np.linspace(0,len(obs_x),len(obs_x))  
        colors = ['r','g','b','c','m','y']
                 
        plt.figure(1)
        k=0
        for target_var in plot_vars:
            j = self.feat_index_dict[target_var]
            plt.subplot(len(plot_vars),1,k+1)
            
            plt.plot(timeline,obs_x[:,j],colors[j%6]+"-",label=target_var)
            plt.plot(timeline, recon_x[:,j], "k-",label='reconstructed')
            plt.legend(loc='best', prop={'size': 6})
            k+=1
        
        plt.show()
    
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tempfile
from .. import BaseModel
import random

def oneclass_loss(z,radius,nu):
    dist = tf.reduce_sum(tf.square(z), axis=-1)
    loss = tf.maximum(dist - radius ** 2, tf.zeros_like(dist))
    loss =  radius**2+(1/nu)*tf.reduce_mean(loss)
    return loss

class DeepSVDD(BaseModel):

    def __init__(self, signals):
        self.signals = signals
        
        self.targets = []
        for signal in self.signals:
            if signal.isInput and signal.isOutput:
                self.targets.append(signal.name)
    
    def score_samples(self, x):
        z = self._model.predict(x)
        dists = []
        for t in range(len(x)):
            z_t = z[t]
            dist = np.sum(np.square(z_t))
            dists.append(dist)
        
        return np.array(dists)
    
    
    def predict(self,x):
        z = self.estimator.predict(x)
        return z
    
    def _get_train_fn(self):
        @tf.function
        def _train_step(x,model,radius,nu,optimizer):
            with tf.GradientTape() as tape:
                z = model(x)
                loss = oneclass_loss(z,radius,nu)
            gradients = tape.gradient(loss, model.trainable_variables+[radius])
            optimizer.apply_gradients(zip(gradients, model.trainable_variables+[radius]))   
            return loss
        return _train_step
    
    
    
    def train(self, x, z_dim, nu=0.1, hidden_layers=1, z_activation='tanh', batch_size=256,epochs=10, verbose=0):
        np.random.seed(123)
        random.seed(1234)
        tf.random.set_seed(1234)
        keras.backend.clear_session()
        model = self._make_network(x.shape[1], z_dim, hidden_layers,z_activation)
        # model.summary()
        
        radius = tf.Variable(0.1, dtype=np.float32)
        optimizer = tf.keras.optimizers.Adam()
        train_fn = self._get_train_fn()
        
        verbose_interval = epochs//10
        for ep in range(epochs):
            shuffle_index = np.arange(len(x))
            np.random.shuffle(shuffle_index)
            x = x[shuffle_index]
            
            ep_loss = 0
            iter_num = int(x.shape[0]//batch_size)
            for i in range(iter_num):
                batch_x = x[i*batch_size:(i+1)*batch_size].astype(np.float32)
                loss = train_fn(batch_x,model,radius,nu,optimizer)
                ep_loss += loss
            if verbose and ep % verbose_interval == 0:
                print('epoch:',ep,'/',epochs)
                print('loss',ep_loss/iter_num)
                print('radius',radius)
                print('dloss',loss)
                print()
        self._model = model
        return self
    
    
    def score(self,neg_x,neg_y):
        """
        Score the model based on datasets with uniform negative sampling.
        Better score indicate a higher performance
        For efficiency, the best f1 score of NSIBF-PRED is used for scoring in this version.
        """
        
        pass
    
    def save_model(self,model_path=None):
        """
        save the model to files
        
        :param model_path: the target folder whether the model files are saved (default is None)
            If None, a tempt folder is created
        """
        
        if model_path is None:
            model_path = tempfile.gettempdir()
        
        self.estimator.save(model_path+'/OCAE.h5',save_format='h5')
    
    def load_model(self,model_path=None):
        """
        load the model from files
        
        :param model_path: the target folder whether the model files are located (default is None)
            If None, load models from the tempt folder
        :return self
        """
        if model_path is None:
            model_path = tempfile.gettempdir()
        self.estimator = keras.models.load_model(model_path+'/OCAE.h5')
        return self
    
    
    def _make_network(self, x_dim, z_dim, hidden_layers,z_activation='relu'):
        hidden_dims = []
        interval = (x_dim-z_dim)//(hidden_layers+1)
        
        x_input = keras.Input(shape=(x_dim),name='x_input')
        for i in range(hidden_layers):
            hid_dim = max(1,x_dim-interval*(i+1))
            hidden_dims.append(hid_dim)
            if i == 0:
                g_dense = keras.layers.Dense(hid_dim, activation='relu') (x_input)
            else:
                g_dense = keras.layers.Dense(hid_dim, activation='relu') (g_dense)
        z_out = keras.layers.Dense(z_dim, activation=z_activation,name='z_output')(g_dense)
        model = keras.Model(x_input,z_out)
        return model
    
    
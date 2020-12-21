import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os.path as path
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from sklearn.metrics import mean_squared_error
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from .metrics import asymmetric_mse


class NN_Regressor:
    def __init__(self, num_hidden_layers=3,hidden_unit=100):
        self.curr_path = path.abspath(__file__) # Full path to current class-definition script
        self.root_path = path.dirname(path.dirname(path.dirname(self.curr_path)))
        # specify NN parameters
        # Number of hidden layers
        self.num_hidden_layers=num_hidden_layers
        # Number of units in each hidden layer
        self.hidden_unit=hidden_unit
        self.model = None
    
    def train(self, X, y,activation='relu',num_epoch=20,learning_rate=0.001):
        """Fits the regression model

        Args:
            X (pd.DataFrame or Series): df with features
            y (pd.Series): contains true training return rates
        """
        try:
            input_shape=X.shape[1]
        except:
            input_shape=1
        #Hidden Layer
        #same activation for all hidden layers
        if isinstance(activation,str):
            model=Sequential()
            for i in range(self.num_hidden_layers):
                if i==0:
                    model.add((layers.Dense(self.hidden_unit, activation=activation, input_shape=(input_shape,))))
                else:
                    model.add((layers.Dense(self.hidden_unit, activation=activation)))
        #list of activation funcitons, the length must match number of hidden layers
        else:
            if len(activation)!=self.num_hidden_layers:
                raise ValueError(f'number of activations must match number of hidden layers {self.num_hidden_layers}')
            model=Sequential()
            for i in range(self.num_hidden_layers):
                if i==0:
                    model.add((layers.Dense(self.hidden_unit, activation=activation[i], input_shape=(input_shape,))))
                else:
                    model.add((layers.Dense(self.hidden_unit, activation=activation[i])))
        # Output layer
        model.add((layers.Dense(1, activation='sigmoid')))

        # Free up memory
        K.clear_session()
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'
        )
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch <20:
                return lr * 0.3
            else:
                return lr*0.1
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks = [lr_scheduler,early_stopping]
        # compile and train model
        model.compile(
            optimizer=optimizer,
            loss=asymmetric_mse,
            metrics=[metrics.RootMeanSquaredError()])
        print('Start training model')
        start=time.time()
        history=model.fit(X,y, 
                          epochs=num_epoch, 
                          batch_size=128,
                          callbacks=callbacks,
                          validation_split=0.2, 
                          verbose=0)
        end=time.time()
        print(f'Succesfully trained model, time used {end-start}')
        self.model=model
        
    def save_model(self,filename):
        """Save the regression model

        Args:
            filename (str) : location to save model
        """
        if self.model is None:
            raise Exception("No trained model to save")
        path=os.path.join(self.root_path,'models/'+filename+'.hdf5')
        self.model.save(path)
        print(f'succesfully saved model to {path}')

    def load_model(self, filename):
        """Loads a pre-trained model (hdf5 model)

        Args:
            filename (str): location of model to load
        """
        filename = path.join(self.root_path, 'models/'+filename+'.hdf5')
        self.model = tf.keras.models.load_model(filename,compile=False)
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss=asymmetric_mse,
            metrics=[metrics.RootMeanSquaredError()])
        print(self.model.summary())
        print(f'Succesfully load model from {filename}')

    def predict(self, X):
        """Regression model predicts on new dataset
   
        Args:
            X (pd.DataFrame or Series): df with features to predict on
        """
        if self.model is None:
            raise Exception("No trained model for prediction")
        return self.model.predict(X)
    
    def evaluate(self,X,y):
        """Evaluate model performance on dataset
   
        Args:
            X (pd.DataFrame or Series): df with features to predict on
            y :target
        """
        if self.model is None:
            raise Exception("No trained model for evaluation")
        return (np.sqrt(self.model.evaluate(X,y)[0]))
 

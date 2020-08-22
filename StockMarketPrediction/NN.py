# CS521: Statistical Natural Language Processing
# University of Illinois at Chicago
# Spring 2020
# Stock Market Prediction using Deep Learning
# =========================================================================================================

from keras.layers import Dense, Embedding, Input, Flatten, Conv1D, MaxPooling1D, LSTM 
from keras.layers.core import Dropout
from keras.models import Model 
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import History 
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from utils import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from tensorflow import keras
from keras.models import Sequential

# Function to build the model and evaluate the model
# Arguments:
# word_index: A dict of all the unique tokens
# X_train: A list of training samples
# y_train: A label of training samples 
# X_val: A list of test samples
# y_val: A label of test samples

def build( word_index, X_train, y_train, X_val, y_val):    
    MAX_SEQUENCE_LENGTH = 1000
    EMBEDDING_DIM = 300
    DROPOUT_RATE = 0.4
    INNERLAYER_DROPOUT_RATE = 0.2
    
    np.random.seed()
    # load embeddings into a dict
    embeddings_index = load_glove()
   
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
     
    
#     embedding_layer = Embedding(len(word_index) + 1,
#                                 EMBEDDING_DIM,
#                                 input_length=MAX_SEQUENCE_LENGTH)
#     
    # use glove embeddings as input
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
     
    embedded_sequences = embedding_layer(sequence_input)
    
    # construct model
    x = Conv1D(256, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Dropout(INNERLAYER_DROPOUT_RATE)(x)
     
    x = Conv1D(256, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(INNERLAYER_DROPOUT_RATE)(x)
     
    x = Conv1D(256, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
     
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x) 
    x = Dropout(DROPOUT_RATE)(x)
      
    preds = Dense(3, activation='softmax')(x) 
    
    model = Model(sequence_input, preds)

    print(model.summary())
    
    # stop early to prevent overfitting
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    history = History()

    # compile, fit & evaluate the model 
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer= 'adam',
                metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=12, callbacks = [earlyStopping])
    
    score = model.evaluate(X_val, y_val, verbose=0)
    
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    print(history.history)
    
    plt.figure(1)  
    
    # plot the model  
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()
       
    plt.subplot(211)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()
    
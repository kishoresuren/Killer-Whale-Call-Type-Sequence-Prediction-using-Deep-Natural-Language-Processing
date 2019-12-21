# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:54:38 2019

@author: Kishore
"""

from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

def createUnidirectionalLSTMModel(vocab_size , seq_len):
    print('Selected model is Unidirectional LSTM')
    model = Sequential()
    model.add(LSTM(256,input_shape=(seq_len,1),return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(vocab_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    model.summary()
    return model

def createBidirectionalLSTMModel(vocab_size , seq_len):
    print('Selected model is Bidirectional LSTM')
    model = Sequential()
    model.add(Bidirectional(LSTM(256, input_shape=(seq_len, 1),return_sequences=True)))
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(vocab_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

def createUnidirectionalGRUModel(vocab_size , seq_len):
    print('Selected model is Unidirectional GRU')
    model = Sequential()
    model.add(GRU(256,input_shape=(seq_len,1),return_sequences=True))
    model.add(GRU(128,return_sequences=True))
    model.add(GRU(64))
    model.add(Dense(vocab_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    model.summary()
    return model


def createBidirectionalGRUModel(vocab_size , seq_len):
    print('Selected model is Bidirectional GRU')
    model = Sequential()
    model.add(Bidirectional(GRU(256,input_shape=(seq_len,1),return_sequences=True)))
    model.add(Bidirectional(GRU(128,return_sequences=True)))
    model.add(Bidirectional(GRU(64)))
    model.add(Dense(vocab_size,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model



def fetchData():
    return [np.load('XTrainSet.npy') , np.load('XTestSet.npy') ,np.load('YTrainSet.npy'), np.load('YTestSet.npy')]

def tokenize(list,tokenizer):
    tokenizer.fit_on_texts(list)
    temp = tokenizer.texts_to_sequences(list)
    temp1=[]
    for arr in temp:
        temp1.append([val-1 for val in arr])
    return temp1

def selectModel(modelType,modelName,nClusters):
    if modelType == 'Uni':
        if modelName == 'GRU':
            model = createUnidirectionalGRUModel(nClusters,5)
        else:
            model = createUnidirectionalLSTMModel(nClusters,5)
    else :
        if modelName == 'GRU':
            model = createBidirectionalGRUModel(nClusters,5)
        else:
            print('Bi-LSTM model selected')
            model = createBidirectionalLSTMModel(nClusters,5)

    return model


parser = argparse.ArgumentParser(description='Building a neural network')
parser.add_argument('--numClusters', default='21',required=False,
                    help='The number of unique sound patterns in the cluster file')
parser.add_argument('--modelType', default='Bi',required=False,
                    help='Can be Uni for unidirecyional model or Bi for bidirectional model')
parser.add_argument('--modelName', default='LSTM',required=False,
                    help='Can be LSTM or GRU')

args = parser.parse_args()
   
nClusters = int(args.numClusters)
print(nClusters)
tokenizer = Tokenizer() 

#Tokenize the data and one hot encode the labels  
XTrain = np.array(tokenize(fetchData()[0].tolist(),tokenizer))
XTest = np.array(tokenize(fetchData()[1].tolist(),tokenizer))
YTrain = to_categorical(np.array(tokenize(fetchData()[2].tolist(),tokenizer)))
YTest = to_categorical(np.array(tokenize(fetchData()[3].tolist(),tokenizer)))




modelType = args.modelType
modelName = args.modelName
#Select model based on the input parameters passed. Default: Bi-LSTM model
model = selectModel(modelType ,modelName,nClusters)
XTrain = XTrain.reshape(XTrain.shape[0],XTrain.shape[1],1)
mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

history = model.fit(XTrain,YTrain, batch_size=32,epochs=100,callbacks=[es,mc])

#Plot the accuracies and losses
plt.figure()
plt.xlabel('Epochs')
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
#plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
#plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.savefig('accuracies'+args.modelType+args.modelName+'.png')

#Save the model
model.save(args.modelType+args.modelName+'.h5')

#Save the tokenizer
with open('orcaTokenizer', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
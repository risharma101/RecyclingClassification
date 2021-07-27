# Gathering info on the trained model and running on test dataset
from __future__ import print_function
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
#import batchExtraction
import os
import numpy as np
import tensorflow as tf
import sys

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


if len(sys.argv) == 2 and os.path.exists('splitData/split_' + sys.argv[1].split(".")[0]):
    dataFolder = 'split_' + sys.argv[1].split(".")[0]
else:
    print("Error: expecting python nn_validation.py [pca_file_name]")
    sys.exit()

def loadData():
    loadPath = 'splitData/' + dataFolder
    trainingData = np.load('./{0}/training_data.npy'.format(loadPath))
    trainingLabels = np.load('./{0}/training_labels.npy'.format(loadPath))
    validationData = np.load('./{0}/validation_data.npy'.format(loadPath))
    validationLabels = np.load('./{0}/validation_labels.npy'.format(loadPath))

    return trainingData, trainingLabels, validationData, validationLabels


def modelValidationInfo():
    _, _, validationData, validationLabels = loadData()
    
    validationData = tf.convert_to_tensor(validationData, dtype=tf.float32)

    sess = tf.Session()
    with sess.as_default():

        #define the model
        model = Sequential()
        model.add(Dense(units=100,activation='tanh', input_dim=1223)) # This input_dim needs to be changed based on what data is being fed in
        model.add(Dropout(0.5))
        model.add(Dense(units=100,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(units=6,activation='tanh'))



        # Now load in the saved weights
        model.load_weights('./model/cnn_model.ckpt')
        
        # Evaluate the model and get the predicted labels
        predictions = model(validationData)
        softmax = tf.nn.softmax(predictions)
        
       # predictions_np = predictions.np()
       # softmax_np = softmax.np()

        predicted_label = tf.argmax(softmax, axis=1, output_type=tf.int32).eval()
        print("Prediction: {}".format(predicted_label))
        print("    Labels: {}".format(validationLabels))

        print(confusionMatrix(predicted_label, validationLabels))

        modelMetrics(predicted_label, validationLabels)

def confusionMatrix(predicted, actual):
    correctCount = [0, 0, 0, 0, 0, 0] # each position represents a label
    incorrectCount = [0, 0, 0, 0, 0, 0] # each position represents a label
    for x in range(len(predicted)):
        if (predicted[x] == actual[x]):
            correctCount[predicted[x]] += 1
        else:
            incorrectCount[predicted[x]] += 1 

    return correctCount, incorrectCount

def modelMetrics(predicted,actual):
    print(classification_report(actual, predicted))
    print(confusion_matrix(actual, predicted))
    

    #sns.heatmap(confusion_matrix(actual,predicted))
    #plt.show()


modelValidationInfo()

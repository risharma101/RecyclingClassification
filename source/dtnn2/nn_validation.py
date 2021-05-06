# Gathering info on the trained model and running on test dataset
import numpy as np
import tensorflow as tf

def modelValidationInfo():
    _, _, validationData, validationLabels = batchExtraction.loadData()
    
    #define the model
    model = Sequential()
    model.add(Dense(units=100,activation='tanh', input_dim=4)) # This input_dim needs to be changed based on what data is being fed in
    model.add(Dropout(0.5))
    model.add(Dense(units=100,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=6,activation='tanh'))

    # Now load in the saved weights
    model.load_weights('./model/cnn_model')
    
    # Evaluate the model and get the predicted labels
    predictions = model(validationData)
    softmax = tf.nn.softmax(predictions)
    
    predictions_np = predictions.numpy()
    softmax_np = softmax.numpy()

    predicted_label = tf.argmax(softmax, axis=1)
    print("Prediction: {}".format(predicted_label[:5]))
    print("    Labels: {}".format(validationLabels[:5]))

    print(confusionMatrix(predicted_label, validationLabels))

def confusionMatrix(predicted, actual):
    correctCount = [0, 0, 0, 0, 0, 0] # each position represents a label
    incorrectCount = [0, 0, 0, 0, 0, 0] # each position represents a label
    for x in range(len(predicted)):
        if (predicted[x] == actual[x]):
            correctCount[predicted[x]] += 1
        else:
            incorrectCount[predicted[x]] += 1 

    return correctCount, incorrectCount

modelValidationInfo()
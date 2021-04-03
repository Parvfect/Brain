"""
The goal here is to create a classifier through training a model in the data available through emotions.csv
https://www.kaggle.com/gcdatkin/eeg-emotion-prediction
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.config.list_physical_devices('GPU')

data = pd.read_csv("archive/emotions.csv")
newModel = True
#Access a group of rows and columns by label(s) or a boolean array.

sample = data.loc[0,'fft_0_b':'fft_749_b']
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
#plt.show()


label_mapping = {"POSITIVE" : 2, "NEGATIVE" : 0,"NEUTRAL" : 1}

def preprocess_inputs(df):
    # Divide the data into test and train sets, replace strings of positive etc with numbers
    df = df.copy()

    df['label'] = df['label'].replace(label_mapping) # Replacing strings with 0 - 1 - 2

    y = df['label'].copy()
    X = df.drop('label', axis = 1).copy() #Columnwise

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 123)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)


# Modelling


if(newModel):
    
    newModel = True
    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    expand_dims = tf.expand_dims(inputs, axis=2)

    gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)

    flatten = tf.keras.layers.Flatten()(gru) # Making it one dimensional - 2x2 - 4

    outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile (
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

else:
    model = tf.keras.models.load_model("emotionModel")

def train():
    if(newModel):
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            batch_size=32,
            epochs=50,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        model.save("emotionModel")

def test():
    model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Accuracy: {:.3f}%".format(model_acc * 100))

    cm = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
    plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
    plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)


train()
test()
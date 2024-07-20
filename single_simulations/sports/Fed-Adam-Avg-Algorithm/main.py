import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# PARAMETERS
NUM_ROUNDS = 10
NUM_CLIENTS = 10
BATCH_SIZE = 10
EPOCHS_PER_ROUND = 5
LEARNING_RATE = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

# # SERVER ADAM OPTIMIZER 
fedadam = tf.optimizers.Adam(learning_rate=LEARNING_RATE,beta_1=BETA_1,beta_2 = BETA_2,epsilon=EPSILON)

def rate(r):
    ary2 = []
    for rating in r:
        tv = [0,0,0,0,0]
        tv[rating-1] = 1
        ary2.append(tv)
    return np.array(ary2)



df = pd.read_csv("../processed_data/final_pre-processed.csv")
df = df.drop('Unnamed: 0',axis=1)


df["Ratings"] = df["Ratings"].astype(int)
y = rate(df['Ratings'].values)

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'],y,test_size=.2)



text_vectorizer = TfidfVectorizer(max_df=.8)
text_vectorizer.fit(df['reviewText'])
X_train = text_vectorizer.transform(X_train)
v = []
for j in X_train:
    v.append([j])
X_train = v
X_test = text_vectorizer.transform(X_test)
v = []
for j in X_test:
    v.append([j])
X_test = v

X_train = np.array(X_train)
X_test = np.array(X_test)


x_train_clients = np.array_split(X_train,NUM_CLIENTS)
y_train_clients = np.array_split(y,NUM_CLIENTS)

x_test_clients = np.array_split(X_test,NUM_CLIENTS)
y_test_clients = np.array_split(y_test,NUM_CLIENTS)

print(X_train.shape,X_test.shape)

global_model = Sequential()
global_model.add(Dense(128,input_dim=X_train.shape[0]))
global_model.add(Dense(5,activation='softmax'))
global_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

fed_model = global_model
fed_model.compile(optimizer=fedadam, loss='categorical_crossentropy',metrics=['accuracy'])
fed_model.fit(X_train,y_train,epochs=5,batch_size=20)



acc = fed_model.evaluate(X_test,y_test)
print(acc)
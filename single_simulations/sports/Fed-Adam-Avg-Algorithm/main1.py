import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# PARAMETERS
NUM_ROUNDS = 10
NUM_CLIENTS = 10
BATCH_SIZE = 10
EPOCHS_PER_ROUND = 5
LEARNING_RATE = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

# Function to convert ratings to one-hot encoding
def rate(r):
    ary2 = []
    for rating in r:
        tv = [0, 0, 0, 0, 0]
        tv[rating-1] = 1
        ary2.append(tv)
    return np.array(ary2)

# Load your data and preprocess it
df = pd.read_csv("../processed_data/final_pre-processed.csv")
df = df.drop('Unnamed: 0', axis=1)
df["Ratings"] = df["Ratings"].astype(int)
y = rate(df['Ratings'].values)

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], y, test_size=0.2)

# Initialize and fit the TF-IDF vectorizer
text_vectorizer = TfidfVectorizer(max_df=0.8)
text_vectorizer.fit(df['reviewText'])

# Transform and convert to dense arrays in batches
batch_size = 1000

# Convert X_train
num_samples_train = X_train.shape[0]
dense_X_train = np.empty((0, len(text_vectorizer.get_feature_names_out())), dtype=float)
for start in range(0, num_samples_train, batch_size):
    end = min(start + batch_size, num_samples_train)
    batch = X_train[start:end]
    batch = text_vectorizer.transform(batch)
    dense_X_train = np.vstack((dense_X_train, batch))

# Convert X_test
num_samples_test = X_test.shape[0]
dense_X_test = np.empty((0, len(text_vectorizer.get_feature_names_out())), dtype=float)
for start in range(0, num_samples_test, batch_size):
    end = min(start + batch_size, num_samples_test)
    batch = X_test[start:end]
    batch = text_vectorizer.transform(batch)
    dense_X_test = np.vstack((dense_X_test, batch))

# Split the data into client-specific subsets
x_train_clients = np.array_split(dense_X_train, NUM_CLIENTS)
y_train_clients = np.array_split(y_train, NUM_CLIENTS)
x_test_clients = np.array_split(dense_X_test, NUM_CLIENTS)
y_test_clients = np.array_split(y_test, NUM_CLIENTS)

# Print the shapes of your data
print(dense_X_train.shape, dense_X_test.shape)

# Continue with the rest of your code...
global_model = Sequential()
global_model.add(Dense(128,input_dim=X_train.shape[0]))
global_model.add(Dense(5,activation='softmax'))
global_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

fed_model = global_model
fed_model.compile(optimizer=fedadam, loss='categorical_crossentropy',metrics=['accuracy'])
fed_model.fit(X_train,y_train,epochs=5,batch_size=20)



acc = fed_model.evaluate(X_test,y_test)
print(acc)
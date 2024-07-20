
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt



NUM_ROUNDS = 10
NUM_CLIENTS = 10
BATCH_SIZE = 10
EPOCHS_PER_ROUND = 5
LEARNING_RATE = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7




# Initialize lists to store accuracy values
accuracy_vs_epochs = []
accuracy_vs_clients = []

df = pd.read_csv("processed_data/final_pre-processed.csv")

df = df.drop('Unnamed: 0',axis=1)[:45134]
df["Ratings"] = df["Ratings"].astype(int)

text_vectorizer = TfidfVectorizer(max_df=.8)
text_vectorizer.fit(df['reviewText'])
def rate(r):
    ary2 = []
    for rating in r:
        tv = [0,0,0,0,0]
        tv[rating-1] = 1
        ary2.append(tv)
    return np.array(ary2)


X = text_vectorizer.transform(df['reviewText']).toarray()
y = rate(df['Ratings'].values)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)





global_model = Sequential()
global_model.add(Dense(128,input_dim=X_train.shape[1]))
global_model.add(Dense(5,activation='softmax'))

global_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])






x_train_clients = np.array_split(X_train,NUM_CLIENTS)
y_train_clients = np.array_split(y_train,NUM_CLIENTS)

x_test_clients = np.array_split(X_test,NUM_CLIENTS)
y_test_clients = np.array_split(y_test,NUM_CLIENTS)





client_models = []
for i in range(NUM_CLIENTS):
    local_model = Sequential()
    local_model.add(Dense(128,input_dim=X_train.shape[1]))
    local_model.add(Dense(5,activation='softmax'))
    local_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    local_model.fit(x_train_clients[i],y_train_clients[i],epochs=EPOCHS_PER_ROUND, batch_size=BATCH_SIZE,verbose=0)
    acc = local_model.evaluate(x_test_clients[i],y_test_clients[i])
    print(acc)
    client_models.append(local_model)
    accuracy_vs_clients.append(acc)





acc = global_model.evaluate(X_test,y_test)
print(acc)

# SERVER ADAM OPTIMIZER 
fedadam = tf.optimizers.Adam(learning_rate=LEARNING_RATE,beta_1=BETA_1,beta_2 = BETA_2,epsilon=EPSILON)


# Performing Federated Averaging
weights = global_model.get_weights()
for i in range(len(weights)):
    for j in range(NUM_CLIENTS):
        client_weights = client_models[j].get_weights()
        weights[i] += client_weights[i]/NUM_CLIENTS

global_model.set_weights(weights)





acc = global_model.evaluate(X_test,y_test)
print(acc)





# Train the Fed AVG model
fed_model = global_model
fed_model.compile(optimizer=fedadam, loss='categorical_crossentropy',metrics=['accuracy'])
fed_model.fit(X_train,y_train,epochs=5,batch_size=20)

acc = fed_model.evaluate(X_test,y_test)
print(acc)

# Define the number of clients and epochs
num_clients = 10
num_epochs = 5

# Simulate federated learning rounds
for epoch in range(num_epochs):
    # Compute accuracy of the global model
    loss, accuracy = global_model.evaluate(X_test, y_test)  # Replace with your test data
    accuracy_vs_epochs.append(accuracy)

# Plot accuracy versus epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), accuracy_vs_epochs, marker='o')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Epochs')
plt.grid(True)
plt.show()

# Plot accuracy versus clients
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs * num_clients + 1), accuracy_vs_clients, marker='o')
plt.xlabel('Number of Clients')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Clients')
plt.grid(True)
plt.show()


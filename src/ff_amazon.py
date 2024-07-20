import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf




class FederatedTrainFf:
    def __init__(self, X_train, y_train, X_test, y_test, args):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.args = args
        self.global_model = self.build_model()
        self.client_models = []
        self.accuracy_vs_epochs = []
        self.accuracy_vs_clients = []
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.X_train.shape[1]))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model
    
    def train_client_models(self):
        x_train_clients = np.array_split(self.X_train, self.args.num_clients)
        y_train_clients = np.array_split(self.y_train, self.args.num_clients)
        x_test_clients = np.array_split(self.X_test, self.args.num_clients)
        y_test_clients = np.array_split(self.y_test, self.args.num_clients)
        
        for i in range(self.args.num_clients):
            local_model = Sequential()
            local_model.add(Dense(128, input_dim=self.X_train.shape[1]))
            local_model.add(Dense(5, activation='softmax'))
            local_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            local_model.fit(x_train_clients[i], y_train_clients[i], epochs=self.args.num_epochs, batch_size=self.args.batch_size, verbose=0)
            acc = local_model.evaluate(x_test_clients[i], y_test_clients[i])
            print(f'Client {i} accuracy: {acc}')
            self.client_models.append(local_model)
            self.accuracy_vs_clients.append(acc)
    
    def federated_averaging(self):
        weights = self.global_model.get_weights()
        for i in range(len(weights)):
            for j in range(self.args.num_clients):
                client_weights = self.client_models[j].get_weights()
                weights[i] += client_weights[i] / self.args.num_clients
        self.global_model.set_weights(weights)
    
    def train_federated_model(self):
        fedadam = tf.optimizers.Adam(learning_rate=self.args.lr)
        self.global_model.compile(optimizer=fedadam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.global_model.fit(self.X_train, self.y_train, epochs=5, batch_size=20)
        acc = self.global_model.evaluate(self.X_test, self.y_test)
        print(f'Federated model accuracy: {acc}')
    
    def federated_learning(self):
        self.train_client_models()
        self.federated_averaging()
        self.train_federated_model()
        
        # Simulate federated learning rounds
        for epoch in range(self.args.num_rounds):
            loss, accuracy = self.global_model.evaluate(self.X_test, self.y_test)
            self.accuracy_vs_epochs.append(accuracy)
        
        self.plot_results()
    
    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.args.num_rounds + 1), self.accuracy_vs_epochs, marker='o')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Epochs')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.args.num_rounds * self.args.num_clients + 1), self.accuracy_vs_clients, marker='o')
        plt.xlabel('Number of Clients')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Clients')
        plt.grid(True)
        plt.show()


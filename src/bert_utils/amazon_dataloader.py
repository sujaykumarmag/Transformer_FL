

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.bert_utils.data_loader import LocalDataLoader

class DataLoaderFf():
    def __init__(self, args):
        if args.dataset == "software":
            self.df = pd.read_csv("datasets/software.csv")
        elif args.dataset == "sports":
            self.df = pd.read_csv("datasets/sports.csv")[:45134]
        else:
            print(f"No such datasets {args.dataset} available for bert")
            exit()
        self.df = self.df.drop('Unnamed: 0', axis=1)
        self.df["Ratings"] = self.df["Ratings"].astype(int)
        self.text_vectorizer = TfidfVectorizer(max_df=.8)
        self.text_vectorizer.fit(self.df['reviewText'])
        
    def rate(self, r):
        ary2 = []
        for rating in r:
            tv = [0, 0, 0, 0, 0]
            tv[rating - 1] = 1
            ary2.append(tv)
        return np.array(ary2)
    
    def load_data(self):
        X = self.text_vectorizer.transform(self.df['reviewText']).toarray()
        y = self.rate(self.df['Ratings'].values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        return X_train, X_test, y_train, y_test




class DataLoaderBert:
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == "software":
            self.df = pd.read_csv("datasets/software.csv")
        elif args.dataset == "sports":
            self.df = pd.read_csv("datasets/sports.csv")
        else:
            print(f"No such datasets {args.dataset} available for bert")
            exit()
        self.batch_size = args.batch_size
        self.num_clients = args.num_clients
        self.preprocess_data()
        self.split_data()

    def preprocess_data(self):
        test_Ratings = self.df["Ratings"]
        rate = [1 if int(i) >= 3 else 0 for i in test_Ratings]
        if self.dataset == "sports":
            self.df["Ratings"] = rate
        self.df = self.df.drop('Unnamed: 0', axis=1)
        self.df["Ratings"] = self.df["Ratings"].astype(int)
        print(self.df["Ratings"].value_counts())

        # Testing should be done on new data
        self.df, self.test_new = train_test_split(self.df, test_size=0.25, random_state=42)


    def split_data(self):
        self.df_clients = np.array_split(self.df, self.num_clients)
    
    def get_loaders(self):
        return LocalDataLoader(self.test_new, self.batch_size).masking()
    
    def get_client_data(self):
        return self.df_clients

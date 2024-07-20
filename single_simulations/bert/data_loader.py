from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cpu")


class LocalDataLoader():

    def __init__(self,df,batch_size):
        self.train_text, self.temp_text, self.train_labels, self.temp_labels = train_test_split(df['reviewText'], df['Ratings'],random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=df['Ratings'])
        self.val_text, self.test_text, self.val_labels, self.test_labels = train_test_split(self.temp_text, self.temp_labels, random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=self.temp_labels)
        
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.batch_size = batch_size




    def tokenize(self):
        # tokenize and encode sequences in the training set
        self.tokens_train = self.tokenizer.batch_encode_plus(
            self.train_text.tolist(),
            max_length=25,
            pad_to_max_length=True,
            truncation=True
        )
        # tokenize and encode sequences in the validation set
        self.tokens_val = self.tokenizer.batch_encode_plus(
            self.val_text.tolist(),
            max_length = 25,
            pad_to_max_length=True,
            truncation=True
        )
        # tokenize and encode sequences in the test set
        self.tokens_test = self.tokenizer.batch_encode_plus(self.test_text.tolist(),max_length = 25,pad_to_max_length=True,
                                                  truncation=True)
        
        
    

    def masking(self):
        self.tokenize()

        train_seq = torch.tensor(self.tokens_train['input_ids'])
        train_mask = torch.tensor(self.tokens_train['attention_mask'])
        train_y = torch.tensor(self.train_labels.tolist())

        val_seq = torch.tensor(self.tokens_val['input_ids'])
        val_mask = torch.tensor(self.tokens_val['attention_mask'])
        val_y = torch.tensor(self.val_labels.tolist())
        
        test_seq = torch.tensor(self.tokens_test['input_ids'])
        test_mask = torch.tensor(self.tokens_test['attention_mask'])
        test_y = torch.tensor(self.test_labels.tolist())


        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=self.batch_size)

        return train_dataloader,val_dataloader,test_mask,test_seq,test_y
    

    



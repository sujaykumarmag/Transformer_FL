import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast, AdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

from src.bert_utils.data_loader import LocalDataLoader



class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
      
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
      
        # apply softmax activation
        x = self.softmax(x)

        return x
    
    








class FederatedTrainBERT:
    def __init__(self, data_loader, args):
        self.device = torch.device("cpu")
        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.batch_size = args.batch_size
        self.epochs_per_round = args.num_epochs 
        self.data_loader = data_loader
        
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.test_loader = self.data_loader.get_loaders()
        self.clients_data = self.data_loader.get_client_data()
        
        self.global_model = BERT_Arch(self.bert).to(self.device)
        self.g_acc = []
        self.g_mae = []
        
    def train(self, train_dataloader, model, cross_entropy, optimizer):
        model.train()
        total_loss = 0
        total_preds = []
        
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
                
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch
            
            model.zero_grad()
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
        
        avg_loss = total_loss / len(train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        
        return avg_loss, total_preds
    
    def evaluate(self, val_dataloader, model, cross_entropy):
        print("\nEvaluating...")
        model.eval()
        total_loss = 0
        total_preds = []
        
        for step, batch in enumerate(val_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
                
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch
            
            with torch.no_grad():
                preds = model(sent_id, mask)
                loss = cross_entropy(preds, labels)
                total_loss += loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)
        
        avg_loss = total_loss / len(val_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        
        return avg_loss, total_preds
    
    def federated_rounds(self, round_num):
        client_weights = []
        client_losses = []
        client_acc = []
        client_mae = []
        client_train_losses = []
        client_classification = []
        
        for i in range(self.num_clients):
            decentral_data = self.clients_data[i]
            loader = LocalDataLoader(decentral_data, self.batch_size)
            train_loader, val_loader, test_mask, test_seq, test_y = loader.masking()
            
            for param in self.bert.parameters():
                param.requires_grad = False
                
            model = BERT_Arch(self.bert).to(self.device)
            optimizer = AdamW(model.parameters(), lr=1e-5)
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(loader.train_labels), y=loader.train_labels)
            weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            cross_entropy = nn.NLLLoss(weight=weights)
            
            best_valid_loss = float('inf')
            train_losses = []
            valid_losses = []
            
            for epoch in range(self.epochs_per_round):
                print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs_per_round))
                train_loss, _ = self.train(train_loader, model, cross_entropy, optimizer)
                valid_loss, _ = self.evaluate(val_loader, model, cross_entropy)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    
                    torch.save(model.state_dict(), f'saved_weights_{i+1}.pt')
                    
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                
                print(f'\nTraining Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')
            
            with torch.no_grad():
                preds = model(test_seq.to(self.device), test_mask.to(self.device))
                preds = preds.detach().cpu().numpy()
                
            preds = np.argmax(preds, axis=1)
            client_train_losses.append(train_losses)
            client_losses.append(valid_losses)
            client_weights.append(model.state_dict())
            client_acc.append(accuracy_score(test_y, preds))
            client_classification.append(classification_report(test_y, preds))
            client_mae.append(mean_absolute_error(test_y, preds))
            
        data = list(zip(client_acc, client_train_losses, client_losses, client_mae, client_classification))
        df = pd.DataFrame(data, columns=['Accuracy', 'Train Losses', 'Test Losses', 'MAE', 'Classification'])

        df.to_csv(f'results_{round_num}.csv', index=False)
        
        self.plot_metrics(client_acc, client_losses, client_train_losses, round_num)
        
        return client_weights
    
    def plot_metrics(self, client_acc, client_losses, client_train_losses, round_num):
        num_clients = list(range(1, self.num_clients + 1))
        epochs = range(1, len(client_losses) + 1)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        
        ax1.plot(num_clients, client_acc, marker='o', linestyle='-', color='b')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Clients Accuracy')
        ax1.grid(True)
        
        ax2.plot(epochs, client_losses, marker='o', linestyle='-', color='b', label='Loss')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss vs Accuracy')
        ax2.grid(True)
        
        ax3.plot(epochs, client_acc, marker='o', linestyle='-', color='r', label='Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True)
        
        ax2.legend()
        ax3.legend()
        
        plt.savefig(f"combined_plot_{round_num}.png")
        
    def g_evaluate(self):
        with torch.no_grad():
            preds = self.global_model(self.test_loader[1].to(self.device), self.test_loader[2].to(self.device))
            preds = preds.detach().cpu().numpy()
            
        preds = np.argmax(preds, axis=1)
        print(classification_report(self.test_loader[3], preds))
        print(mean_absolute_error(self.test_loader[3], preds))
        
        return accuracy_score(self.test_loader[3], preds), mean_absolute_error(self.test_loader[3], preds)
    
    def federated_learning(self):
        for i in range(self.num_rounds):
            local_parameters = list(self.global_model.parameters())
            client_weights = self.federated_rounds(i)
            
            for j in range(self.num_clients):
                model = BERT_Arch(self.bert)
                model.load_state_dict(client_weights[j])
                
                for param1, param2 in zip(local_parameters, model.parameters()):
                    param1.data += param2.data
                    
            for params in local_parameters:
                params.data /= self.num_clients
                
            for param1, avg_param in zip(self.global_model.parameters(), local_parameters):
                param1.data = avg_param.data
                
            acc, mae = self.g_evaluate()
            self.g_acc.append(acc)
            self.g_mae.append(mae)
            self.plot_global_metrics()


    def plot_global_metrics(self):
        num_rounds = list(range(1, self.num_rounds + 1))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(num_rounds, self.g_acc, marker='o', linestyle='-', color='g', label='Global Accuracy')
        ax.plot(num_rounds, self.g_mae, marker='o', linestyle='-', color='y', label='Global MAE')
        ax.set_xlabel('Num Rounds')
        ax.set_ylabel('Accuracy and MAE')
        ax.grid(True)
        ax.legend()
        plt.savefig("global_metrics.png")

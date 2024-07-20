from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,mean_absolute_error
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight

# Local Imports
from data_loader import LocalDataLoader
from bert import BERT_Arch

# specify GPU
device = torch.device("cpu")

bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the Data
df = pd.read_csv("../processed_data/final_pre-processed.csv")
df = df.drop('Unnamed: 0',axis=1)
df["Ratings"] = df["Ratings"].astype(int)

# function to train the model
def train(train_dataloader,model,cross_entropy,optimizer):
    model.train()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    total_preds=[]
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to('cpu') for r in batch]
        sent_id, mask, labels = batch
        # clear previously calculated gradients 
        model.zero_grad()        
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
    # append the model predictions
    total_preds.append(preds)
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
      # predictions are in the form of (no. of batches, size of batch, no. of classes).
      # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds




# function for evaluating the model
def evaluate(val_dataloader,model,cross_entropy):
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save the model predictions
    total_preds = []
    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        # push the batch to gpu
        batch = [t.to('cpu') for t in batch]
        sent_id, mask, labels = batch
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds






loader = LocalDataLoader(df,100)
train_loader,val_loader,test_mask,test_seq,test_y = loader.masking()

for param in bert.parameters():
    param.requires_grad = False

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
model = model.to('cpu')

optimizer = AdamW(model.parameters(),lr = 1e-5) 
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(loader.train_labels), y= loader.train_labels)
weights= torch.tensor(class_weights,dtype=torch.float)
weights = weights.to('cpu')
# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 
# set initial loss to infinite
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
for epoch in range(10):
    print('\n Epoch {:} / {:}'.format(epoch + 1, 10))
    #train model
    train_loss, _ = train(train_loader,model,cross_entropy,optimizer)
    #evaluate model
    valid_loss, _ = evaluate(val_loader,model,cross_entropy)
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'weights/saved_weights'+str(epoch+1)+'.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

with torch.no_grad():
    preds = model(test_seq.to('cpu'), test_mask.to('cpu'))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))
print(mean_absolute_error(test_y,preds))

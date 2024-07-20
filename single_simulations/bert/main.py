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

# Load the Data
df = pd.read_csv("final_pre-processed1.csv")[:12805*4]
test_Ratings = df["Ratings"]
rate = []
for i in test_Ratings:
    if int(i) >=3:
        rate.append(1)
    else:
        rate.append(0)
df["Ratings"] = rate
df = df.drop('Unnamed: 0',axis=1)
df["Ratings"] = df["Ratings"].astype(int)

# Testing should be done on new data
test_new = df.tail(12000)
df = df[:-12000]

# PARAMETERS
NUM_ROUNDS = 1
NUM_CLIENTS = 10
BATCH_SIZE = 256
EPOCHS_PER_ROUND = 10
LEARNING_RATE = 0.1
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

loader = LocalDataLoader(test_new,256)
train_loader,val_loader,test_mask,test_seq,test_y = loader.masking()

bert = AutoModel.from_pretrained('bert-base-uncased')

df_clients = np.array_split(df,NUM_CLIENTS)


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

     

def Federated_rounds(rounds):
    client_weights = []
    client_losses = []
    client_acc = []
    client_mae = []
    client_train_losses = []
    client_classification = []
    for i in range(0,NUM_CLIENTS):
        decentral_data = df_clients[i]
        loader = LocalDataLoader(decentral_data,BATCH_SIZE)
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
        for epoch in range(EPOCHS_PER_ROUND):
            print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS_PER_ROUND))
            #train model
            train_loss, _ = train(train_loader,model,cross_entropy,optimizer)
            #evaluate model
            valid_loss, _ = evaluate(val_loader,model,cross_entropy)
            #save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'weights/saved_weights'+str(i+1)+'.pt')
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

        with torch.no_grad():
            preds = model(test_seq.to('cpu'), test_mask.to('cpu'))
            preds = preds.detach().cpu().numpy()
        # model's performance
        preds = np.argmax(preds, axis = 1)
        client_train_losses.append(train_losses)
        client_losses.append(valid_losses)
        client_weights.append(model.state_dict())
        client_acc.append(accuracy_score(test_y, preds))
        client_classification.append(classification_report(test_y,preds))
        client_mae.append(mean_absolute_error(test_y,preds))


    data = [client_acc, client_train_losses, client_losses, client_mae,client_classification]
    data = list(map(list, zip(*data)))
    column_names = ['Accuracy', 'Train Losses', 'Test Losses', 'MAE','Classification']
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv('client/csv/results'+str(rounds)+'.csv')

    
    num_clients = list(range(1, NUM_CLIENTS + 1))
    epochs = range(1, len(client_losses) + 1)
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

        
    # Plot client accuracy at that round
    ax1.plot(num_clients, client_acc, marker='o', linestyle='-', color='b')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Clients Accuracy')
    ax1.grid(True)

    # Plot loss values
    ax2.plot(epochs, client_losses, marker='o', linestyle='-', color='b', label='Loss')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Accuracy')
    ax2.grid(True)

    # Plot accuracy values
    ax3.plot(epochs, client_acc, marker='o', linestyle='-', color='r', label='Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True)

    # Add legends
    ax2.legend()
    ax3.legend()

    # Save the combined plot
    plt.savefig("client/client_plots/combined_plot"+str(rounds)+".png")        
        
    return client_weights
    

def g_evaluate(model):
    # get predictions for test data
    with torch.no_grad():
        preds = model(test_seq.to('cpu'), test_mask.to('cpu'))
        preds = preds.detach().cpu().numpy()

    # model's performance
    preds = np.argmax(preds, axis = 1)
    print(classification_report(test_y, preds))
    print(mean_absolute_error(test_y,preds))
    return accuracy_score(test_y,preds),mean_absolute_error(test_y,preds)















# Initialize the global model
global_model = BERT_Arch(bert)
global_model = global_model.to('cpu')
g_acc = []
g_mae = []

for i in range(NUM_ROUNDS):
    local_parameters = global_model.parameters()
    client_weights = Federated_rounds(i)
    # Perform federated averaging
    for j in range(NUM_CLIENTS):
        # Load the weights from the second model file into model2
        path2 = client_weights[j]
        model2 = BERT_Arch(bert)
        model2.load_state_dict(client_weights[j])  # Load the model state_dict from file

        # Update local parameters with the parameters from model2
        for param1, param2 in zip(local_parameters, model2.parameters()):
            param1.data += param2.data

    # Average the parameters across all clients
    for params in local_parameters:
        for param in params:
            param.data /= NUM_CLIENTS  # Use param.data to modify the tensor in-place

    # Update the global model with the averaged parameters
    for param1, avg_params in zip(global_model.parameters(), local_parameters):
        param1.data = avg_params.data

    acc, mae = g_evaluate(global_model)
    g_acc.append(acc)
    g_mae.append(mae)


num_rounds = list(range(1, NUM_ROUNDS + 1))
fig, ax = plt.subplots(figsize=(8, 6))

# Plot global accuracy and MAE
ax.plot(num_rounds, g_acc, marker='o', linestyle='-', color='g', label='Global Accuracy')
ax.plot(num_rounds, g_mae, marker='o', linestyle='-', color='y', label='Global MAE')
ax.set_xlabel('Num Rounds')
ax.set_ylabel('Accuracy and MAE')
ax.grid(True)

# Add legend
ax.legend()

# Save the combined plot
plt.savefig("global_metrics.png")






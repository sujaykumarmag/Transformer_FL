import argparse
from src.bert_utils.amazon_dataloader import DataLoaderBert, DataLoaderFf
from src.bert import FederatedTrainBERT


parser = argparse.ArgumentParser(description="Federated Learning for Transformers (Training Paradigm)")

parser.add_argument("dataset",type=str,default="software")
parser.add_argument("model",type=str,default="bert")

parser.add_argument("--num_epochs",type=int,default=10)
parser.add_argument("--num_clients",type=int,default=10)
parser.add_argument("--num_rounds",type=int,default=1)

parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--device",type=str,default="cpu")


args = parser.parse_args()

print(args)

if args.model == "bert":
    federated_train_bert = FederatedTrainBERT(DataLoaderBert(args),args).federated_learning()
elif args.model == "ff":
    if args.dataset == "sports" or args.dataset == "software":
        X_train, X_test, y_train, y_test =  DataLoaderFf().load_data()
        federated_trainer = FederatedTrainFf(X_train, y_train, X_test, y_test, args).federated_learning()
    elif args.dataset == "movielens":
       import src.ff_amazon
elif args.model == "bst":
    import src.bst

# Transformer based Federated Learning models for Recommendation Systems

Our study introduces an innovative approach that combines the privacy-preserving attributes of federated learning with the advanced capabilities of transformer-based models, specifically tailored for recommendation systems. Federated learning emerges as a decentralized alternative to traditional machine learning, enhancing both user privacy and data security. 

## Abstract

Our research employs two distinct transformer models: 
1. BERT (Bidirectional Encoder Representations from Transformers)
2. BST (Behavior Sequence Transformer), within a federated learning context.

These models performance is analyzed using the Amazon Customer Review dataset and movielens dataset. 


The empirical results are compelling: 
1. The federated BERT model achieves a notable 87% and 76% accuracy in the global model for 2 different datasets.
2. The federated BST model demonstrates a performance with a mean absolute error of 0.8. 

This investigation not only highlights the effectiveness of federated learning in boosting model accuracy but also emphasizes its crucial role in preserving user privacy. 

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [File Structure](#file-structure)

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Architecture

Our system architecture leverages federated learning to train transformer-based models for recommendation systems. We employ BERT, Feed-Forward and BST models to analyze and predict user preferences from the Amazon Customer Review dataset. The architecture is designed to preserve user privacy by decentralizing the training process.

## Usage

### Training the Model

To train the model, use the following command (make sure to download the datasets and assign the path accordingly):

```bash
python train.py --dataser [DATASET_NAME] --model [MODEL_NAME] 
```
Replace `[MODEL_NAME]` with `bert` or `bst` and `[DATASET_PATH]` with the name of the dataset.


## File Structure

```
├── datasets
│   ├── m1-1m
│   ├── sports.csv
│   ├── software.csv
│
│  
├── notebooks
│   ├── example_runs (...)
│
│  
├── single_simulations
│   ├── bert
│   ├── movie_BST
│   ├── software
│   ├── sports
│
│   
├── src
│   ├── bert.py
│   │
│   ├── bst.py
│   │
│   ├── ff_amazon.py
│   │
│   ├── ff_movielens.py
│   │
│   ├── bert_utils
│       ├── amazon_dataloader.py
│       ├── data_loader.py
│
├── results (all results for each simulations)
│ 
│ 
├── train.py
│   
├── README.md
│   
├── requirements.txt
```

This structure outlines the main directories and files necessary for running the project.


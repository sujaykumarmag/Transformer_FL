#!/bin/bash

python3 train.py sports bert
python3 train.py software bert
python3 train.py sports ff
python3 train.py software ff
python3 src/bst.py
python3 src/ff_amazon.py

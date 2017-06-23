#!/usr/bin/env bash

echo "Preprocessing files..."
cd ~/Documents/tensorflow/twitter-sentiment-master/preprocessing
./run_preprocessing.sh

echo "train model..."
cd ~/Documents/tensorflow/twitter-sentiment-master
python3 train_model.py

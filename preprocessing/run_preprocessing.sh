#!/usr/bin/env bash

echo "Will preprocess everything accordingly."
echo "Will trim all tweets to maximum length 35."

./build_vocab.sh
python3 preprocessv2.py --full --sentence-length 35

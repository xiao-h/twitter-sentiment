
import pickle
import gensim
import getopt
import numpy as np
import sys

# FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
# FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
# POS_FILE_NAME = "../data/train/train_pos.txt"
# NEG_FILE_NAME = "../data/train/train_neg.txt"
VALID_FILE_NAME = "../data/test/test_data.txt"
# VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"
# WORD2VEC_FILE_NAME = "../../glove.6B.300d.gensim.txt"
# WORD2VEC_FILE_NAME = "../../glove.6B.50d.gensim.txt"
# WORD2VEC_FILE_NAME = "../../glove.twitter.27B.50d.gensim.txt"

VOCAB_FOLDER = "../data/preprocessing/"
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"

VALID_SIZE = 359

with open(MAPPINGS_FOLDER+"mappings.pkl", 'rb') as f:
    (mappings, pretrained, extra_words) = pickle.load(f)


with open(VOCAB_FOLDER+"full-vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)


def handle_hashtags_and_mappings(line, vocab):
    result_line = []
    for word in line.split():
        if word[0] == '#' and word not in vocab:  # if hashtag but is in vocab then leave it as it is (it has big enough frequency)
            word = word[1:]                       # otherwise split it to normal words
            length = len(word)
            word_result = []
            claimed = np.full(length, False, dtype=bool)  #initially all letters are free to select
            for n in range(length, 0, -1):  #initially search for words with n letters, then n-1,... until 1 letter words
                for s in range(0, length-n+1):  #starting point. so we examine substring  [s,s+n)
                    substring = word[s:s+n]
                    if substring in vocab:
                        if ~np.any(claimed[s:s+n]):   #nothing is claimed so take it
                            claimed[s:s+n] = True
                            word_result.append((s, substring))
            word_result.sort()
            for _, substring in word_result:
                result_line.append(substring)
        else:  # it is not a hashtag. check if it has a mapping (spelling correction)
            if word in mappings:
                result_line.append(mappings[word])
            else:
                result_line.append(word)
    return ' '.join(result_line)


def prepare_valid_data(max_sentence_length, vocab):
    validate_x = np.zeros((VALID_SIZE, max_sentence_length))
    i = 0
    cut = 0
    empty = 0
    with open(VALID_FILE_NAME) as f:
        for tweet in f:
            tweet = tweet.strip()
            # tweet = tweet[6:]   # remove prefix   "<num>,"
            tweet = handle_hashtags_and_mappings(tweet, vocab)
            j = 0
            for word in tweet.split():
                if word in vocab:
                    validate_x[i, j] = vocab[word]
                    j += 1
                if j == max_sentence_length:
                    cut += 1
                    # print("cut: "+line)
                    # cut sentences longer than max sentence lenght
                    break
            if j == 0:
                #print(tweet)
                empty += 1
            i += 1
    print("Preprocessing done. {} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))
    return validate_x



def main():
    print('prepare validation data..')
    max_sentence_length = 35
    validate_x = prepare_valid_data(max_sentence_length, vocab)
    np.save('../data/preprocessing/validateX', validate_x)


if __name__ == "__main__":
    main()
    #test_preprocessing()

# ./preprocessv2.py --full --sentence-length=35

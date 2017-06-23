
import pickle
import gensim
import getopt
import numpy as np
import sys

VOCAB_FILE_NAME = "../data/preprocessing/vocab_cut.txt"
ORD2VEC_FILE_NAME = "../../glove.6B.50d.gensim.txt"
MAPPINGS_FOLDER = "../data/preprocessing/mappings/"

VALID_SIZE = 1600000
FULL_TRAIN_SIZE = 1600000
SMALL_TRAIN_SIZE = 600

# the above will be replaced by data from the DB

with open(MAPPINGS_FOLDER+"mappings.pkl", 'rb') as f:
    (mappings, pretrained, extra_words) = pickle.load(f)


def vocab_and_embeddings(prefix):
    """
    pickle vocabulary
    """
    vocab = dict()
    vocab_inv = dict()
    # pre insert the padding word
    index = 0
    vocab["<PAD/>"] = index
    vocab_inv[index] = "<PAD/>"
    index += 1

    print("We have {0} extra words.".format(len(extra_words)))
    for word in extra_words:
        if word in pretrained:
            print("in extra_words and pretrained simultaneously!: "+word)

    for word in extra_words:
        vocab[word] = index
        vocab_inv[index] = word
        index += 1

    """
    pickle word embeddings
    """
    model =  gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_FILE_NAME, binary=False)
    embedding_dim = len(model['a'])
    X = np.empty( (len(extra_words)+len(pretrained)+1  ,embedding_dim)  )
    X[0:len(extra_words)+1] = np.random.uniform(-0.25, 0.25, size=(len(extra_words)+1, embedding_dim))
    print("create word2vec lookup table..")

    assert index == len(extra_words)+1

    for word in pretrained:
        vocab[word] = index
        vocab_inv[index] = word
        X[index] = model[word]
        index += 1

    # sanity check
    print("len(vocab)= {}".format(len(vocab)))
    print("len(vocab_inv)= {}".format(len(vocab_inv)))
    print("len(extra_words)+len(pretrained)+1= {}".format(len(extra_words)+len(pretrained)+1))
    #assert len(vocab) == (len(extra_words) + len(pretrained) + 1) == len(vocab_inv)

    with open('../data/preprocessing/{0}-vocab.pkl'.format(prefix), 'wb') as f:
        pickle.dump(vocab, f, protocol=2)
    with open('../data/preprocessing/{0}-vocab-inv.pkl'.format(prefix), 'wb') as f:
        pickle.dump(vocab_inv, f, protocol=2)
    print("Vocabulary pickled.")

    np.save('../data/preprocessing/{0}-embeddings'.format(prefix), X)
    print("Embeddings pickled.")
    print("Used {} pre-trained word2vec vectors and {} new random vectors.".format(len(pretrained), len(extra_words)+1))

    return vocab


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


def prepare_valid_data(max_sentence_length, vocab, VALID_SIZE, list_tweets):
    validate_x = np.zeros((VALID_SIZE, max_sentence_length))
    i = 0
    cut = 0
    empty = 0
    with open(VALID_FILE_NAME) as f:
        for tweet in f:
            tweet = tweet.strip()
            tweet = handle_hashtags_and_mappings(tweet, vocab)
            j = 0
            for word in tweet.split():
                if word in vocab:
                    validate_x[i, j] = vocab[word]
                    j += 1
                if j == max_sentence_length:
                    cut += 1
                    break
            if j == 0:
                #print(tweet)
                empty += 1
            i += 1
    print("Preprocessing done. {} tweets cut to max sentence lenght and {} tweets disapeared due to filtering."
          .format(cut, empty))
    return validate_x

def main(argv):
    max_sentence_length = 35
 
    vocab = vocab_and_embeddings('full')
    validate_x = prepare_valid_data(max_sentence_length, vocab)
    np.save('../data/preprocessing/validateX', validate_x)


if __name__ == "__main__":
    main(sys.argv[1:])
    #test_preprocessing()



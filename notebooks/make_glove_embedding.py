import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import numpy as np
import sys
import pickle
import os
import collections


class TfidfEmbedding(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec[next(iter(word2vec))])

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, max_df=0.5, stop_words='english')
        tfidf.fit(X)

        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                for w in words if w in self.word2vec] or
                [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def read_tweets(fname):
    """Read the tweets in the given file."""
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        return [l[2] for l in reader]



def get_embedding(all_words, tokens, vocab, embedding):
    glove_embed = {}
    for word in tokens:
        if word in all_words:
            glove_embed[word] = embedding[vocab.index(word)]
        
    return glove_embed


if __name__ == '__main__':

    print('Loading vocabulary and GloVe embedding...')
    with open('vocabulary_50.pkl', 'rb') as f:
        vocab = pickle.load(f)

    embedding = np.load('embeddings_50.npy')
    all_words = set(vocab)

    print('Loading data...')
    # tweets = os.path.join('prediction_baselinetfidf_rand1_v2.csv')
    tweets = os.path.join('viz_risky_v2.csv')
    sentences = read_tweets(tweets)
    tokens = [token for sent in sentences for token in sent.split()]
    print('Length of the full dataset: ', len(sentences))
    print(sentences[:3])

    print('Word2vec embedding...')
    word2vec = get_embedding(all_words, tokens, vocab, embedding)

    X = []
    for sent in sentences:
        X.append(sent.split())
    y = []
    X = np.array(X)

    print('TfIdf feature extraction...')
    vectorizer = TfidfEmbedding(word2vec)
    vectorizer.fit(X, y)
    features_embed = vectorizer.transform(X)

    print(type(features_embed))

    np.save('tfidf_embeded_risky_topic.npy', features_embed)


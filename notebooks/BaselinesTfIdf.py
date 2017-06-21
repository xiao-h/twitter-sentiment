import os

import numpy as np
from sklearn.linear_model import SGDClassifier
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
import argparse


def read_tweets(fname):
    """Read the tweets in the given file."""
    with open(fname, 'r') as f:
        return [l for l in f.readlines()]


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("{2}: Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores),
              i + 1))
        print("Parameters: {0}".format(score.parameters))
        print("")


def eval_tfidf(X_full, y_full, grid):
    gs = GridSearchCV(SGDClassifier(), grid, cv=5, verbose=True)
    print("Starting grid search...")
    res = gs.fit(X_full, y_full)
    report(res.grid_scores_, n_top=25)

    predY = res.predict(X_full)
    acc = accuracy_score(y_full, predY)
    f1 = accuracy_score(y_full, predY)

    print("Train accuracy: {0}\nTrain F1 score: {1}".format(acc, f1))

    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TfIdf with SGDClassifier", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--save_vocab_and_model', help='''Save tfidf vocabulary and model''', action='store_true')
    parser.add_argument('-p', '--plot', help='''plot roc curve and gain chart''', action='store_true')
    parser.add_argument('-f', '--file', help='''File path''', default= '../data/train')

    args = parser.parse_args()

    print('Loading data...')
    POS_TWEET_FILE = os.path.join(args.file, 'train_pos_full.txt')
    NEG_TWEET_FILE = os.path.join(args.file, 'train_neg_full.txt')

    pos_tweets = read_tweets(POS_TWEET_FILE)
    neg_tweets = read_tweets(NEG_TWEET_FILE)

    sentences = pos_tweets + neg_tweets
    y_full = [+1] * len(pos_tweets) + [-1] * len(neg_tweets)
    print('Size of the dataset:', len(sentences))

    print('TfIdf feature extraction...')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_full = vectorizer.fit_transform(sentences)
    print('feq_term_matrix dim:', X_full.shape)


    grid = {
        'loss': ['modified_huber', 'log'],
        'alpha': [1e-6, 5e-6, 0.00001, 0.00005, 0.0001, 0.0005],
    }

    print('Model fitting...')
    res = eval_tfidf(X_full, y_full, grid)


    if args.save_vocab_and_model:
        print('Saving vocabulary...')
        pickle.dump(vectorizer.vocabulary_, open('vectorizer.pkl', 'wb'))
        print('Saving models...')
        joblib.dump(res, 'BaselinesTfidf.pkl')


    if args.plot:
        pred_tfidf = res.predict_proba(X_full)
        fpr, tpr, _ = roc_curve(np.asarray(y_full), pred_tfidf[:, 1])
        plt.figure()
        plt.plot(fpr, tpr)
        plt.grid(True)
        plt.title('ROC curve')
        plt.show()


        npos = len(pos_tweets)
        isort = np.argsort(-pred_tfidf[:, 1])
        n = pred_tfidf.shape[0]
        tpr = np.empty((n,), dtype='float32')
        for i in range(1, n+1):
            tpr[i-1] = (isort[:i-1] < npos).sum()/npos # becasue positive tweets were read in first
        rpp = np.arange(1/n, 1+1/n, 1/n)

        plt.figure()
        plt.plot(rpp, tpr)
        plt.grid(True)
        plt.title('gain curve')
        plt.show()

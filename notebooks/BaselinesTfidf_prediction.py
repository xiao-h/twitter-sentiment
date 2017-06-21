import random
from pattern_matching import *
import get_oracle as go
import logging
import datetime
import itertools
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import constants as CONSTS
import csv
import pickle
import sys
from sqlalchemy import inspect, text

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

random.seed(1)

# INDUSTRIES = ['financial', 'pharmaceuticals', 'auto', 'oil_gas', 'consumer_goods', 'computer_service']

_, sess, top_topics = go.get_table('VIZ_TOP_TOPICS_PREDICTION')


def iter_feeds_peritem(company, log_every=None):
    """
    Yield plain text of each message

    """
    extracted = 0

    start = datetime.datetime.now()
    nrow = sess.query(top_topics).filter(top_topics.c.company.like(company)).count()
    print('total record:', nrow, 'for company: ', company)

    for u in sess.query(top_topics).filter(top_topics.c.company.like(company)):
        extracted += 1

        if log_every and extracted % log_every == 0 and extracted != 0:
            logging.info("extracting file %i, time elapsed: %s" % (extracted, datetime.datetime.now()-start))


        yield (u.external_id, u.company, matched(u.content))


def chunks(iterable, size=20000):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size-1))



if __name__ == "__main__":

    vocabulary = pickle.load(open('vectorizer.pkl', 'rb'))
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', vocabulary=vocabulary)
    res = joblib.load('BaselinesTfidf.pkl')

    ids = []
    X_full = []
    company_ = []

    output_filepath = 'prediction_baselinetfidf_toptopics.csv'
    with open(output_filepath, 'w') as empty:
        pass

    with open(output_filepath, 'a', encoding='utf8', errors='ignore', newline='') as out:
        writer = csv.writer(out, delimiter=',')
        loaded = 0
        total_loaded = 0
        for company, industry in CONSTS.TAG_INDUSTRY_MAP.items():
#            if company not in ['rbc', 'nik', 'tdb', 'for', 'dia', 'mer','bmw', 'ibm', 'wel']:
            for chunk in chunks(iter_feeds_peritem(company, 1e4)):
                output = list(chunk)
                id_in, company_in, x_in = zip(*output)
                ids = (list(id_in))
                company_ = (list(company_in))
                X_full = (list(x_in))


                X_full = vectorizer.fit_transform(X_full)
                pred_tfidf = res.predict_proba(X_full)

                writer.writerows(zip(company_, ids, pred_tfidf[:, 0], pred_tfidf[:, 1]))
                loaded += len(ids)
                total_loaded += len(ids)
                print('Wrote line for this company: ', loaded, ', total loaded: ', total_loaded)

                ids = []
                X_full = []
                company_ = []

            loaded = 0

    print('Done')

    sess.close()
    engine.dispose()


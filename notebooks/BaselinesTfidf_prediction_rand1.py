import random
from pattern_matching import *
import get_oracle as go
import logging
import datetime
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pickle
import sys
from sqlalchemy import inspect, text

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

random.seed(1)

INDUSTRIES = ['financial', 'pharmaceuticals', 'auto', 'oil_gas', 'consumer_goods', 'computer_service']

def iter_feeds_peritem(industry, log_every=None):
    """
    Yield plain text of each message

    """
    extracted = 0
    external_ids = []
    content_out = []
    matched_out = []

    start = datetime.datetime.now()
    _, sess, social_data = go.get_table('SOCIAL_DATA')
    # need a better way, this is memory inefficient
    nrows = sess.query(social_data).filter(social_data.c.industry.ilike('%'+industry+'%')).\
        filter(social_data.c.type.like('Twitter')).filter(social_data.c.content != None).count()

    # print(inspect(top_topics).columns)
    # need a better way, this is memory inefficient
    print('industry: ', industry, ', nrows = ', nrows)
    start_row = random.randint(nrows//2000, nrows-3000)
    print('starting at row: ', start_row)


    for u in sess.query(social_data).filter(social_data.c.industry.ilike('%'+industry+'%')).\
        filter(social_data.c.type.like('Twitter')).filter(social_data.c.content != None)[start_row:start_row+2000]:
        if log_every and extracted % log_every == 0:
            logging.info("extracting file %i, time elapsed: %s" % (extracted, datetime.datetime.now()-start))
        external_ids.append(u.external_id)
        content_out.append(u.content)
        matched_out.append(matched(u.content))
        extracted += 1

    yield external_ids, content_out, matched_out




if __name__ == "__main__":

    vocabulary = pickle.load(open('vectorizer.pkl', 'rb'))
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', vocabulary=vocabulary)
    res = joblib.load('BaselinesTfidf.pkl')

    ids = []
    contents = []
    X_full = []
    industry_ = []

    output_filepath = 'prediction_baselinetfidf_rand1_v2.csv'

    with open(output_filepath, 'w', encoding='utf8', errors='ignore', newline='') as out:
        writer = csv.writer(out, delimiter=',')

        for industry in INDUSTRIES:
            for id, content, cleaned in iter_feeds_peritem(industry, 500):
                ids.extend(id)
                contents.extend(content)
                X_full.extend(cleaned)
                industry_.extend([industry]*2000)

        assert len(ids) == 12000
        assert len(contents) == 12000
        assert len(X_full) == 12000
        assert len(industry_) == 12000

        X_full = vectorizer.fit_transform(X_full)
        pred_tfidf = res.predict_proba(X_full)

        writer.writerows(zip(ids, industry_, contents, pred_tfidf[:,0], pred_tfidf[:,1]))

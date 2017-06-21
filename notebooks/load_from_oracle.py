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

_, sess, top_topics = go.get_table('VIZ_RISKY_V2')

def iter_feeds_peritem(company, log_every=None):
    """
    Yield plain text of each message

    """
    extracted = 0

    start = datetime.datetime.now()
    # _, sess, social_data = go.get_table('SOCIAL_DATA')
    # # need a better way, this is memory inefficient
    # nrows = sess.query(social_data).filter(social_data.c.industry.ilike('%'+industry+'%')).\
    #     filter(social_data.c.type.like('Twitter')).filter(social_data.c.content != None).count()

    nrow = sess.query(top_topics).filter(top_topics.c.company.like(company)).count()
    print('total record: ', nrow, ' for company: ', company)

    # print(inspect(top_topics).columns)
    # need a better way, this is memory inefficient
    # print('industry: ', industry, ', nrows = ', nrows)
    # start_row = random.randint(nrows//2000, nrows-3000)
    # print('starting at row: ', start_row)

    for u in sess.query(top_topics).filter(top_topics.c.company.like(company)):
        extracted += 1

        if log_every and extracted % log_every == 0 and extracted != 0:
            logging.info("extracting file %i, time elapsed: %s" % (extracted, datetime.datetime.now()-start))

        yield (u.external_id, u.company, matched(u.content.rstrip()), u.content.rstrip(), u.topic)


def chunks(iterable, size=20000):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size-1))

# def grouper(n, iterable, fillvalue=None):
#     iterator = iter(iterable)
#     while True:
#         chunk =



if __name__ == "__main__":

    # vocabulary = pickle.load(open('vectorizer.pkl', 'rb'))
    # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', vocabulary=vocabulary)
    # res = joblib.load('BaselinesTfidf.pkl')

    ids = []
    cleaned = []
    company_ = []
    content_ = []
    topic_ = []


    output_filepath = 'viz_risky_v2.csv'
    with open(output_filepath, 'w') as empty:
        pass

    with open(output_filepath, 'a', encoding='utf8', errors='ignore', newline='') as out:
        writer = csv.writer(out, delimiter=',')
        loaded = 0
        total_loaded = 0
        for company, industry in CONSTS.TAG_INDUSTRY_MAP.items():
            for chunk in chunks(iter_feeds_peritem(company, 1e4)):
                output = list(chunk)
                id_in, company_in, cleaned_in, content_in, topic_in = zip(*output)
                ids = (list(id_in))
                company_ = (list(company_in))
                cleaned = (list(cleaned_in))
                content_ = (list(content_in))
                topic_ = (list(topic_in))

                writer.writerows(zip(ids, company_, cleaned, content_, topic_))
                loaded += len(ids)
                total_loaded += len(ids)
                print('Wrote line for this company: ', loaded, ', total loaded: ', total_loaded)

                ids = []
                cleaned = []
                company_ = []
                content_ = []
                topic_ = []

            loaded = 0

    print('Done')




    # for industry in INDUSTRIES:
    #     for id, content, cleaned in iter_feeds_peritem(industry, 500):
    #         ids.extend(id)
    #         contents.extend(content)
    #         X_full.extend(cleaned)
    #         industry_.extend([industry]*2000)
    #
    # assert len(ids) == 12000
    # assert len(contents) == 12000
    # assert len(X_full) == 12000
    # assert len(industry_) == 12000

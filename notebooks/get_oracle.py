import time
from datetime import datetime
import gbl
import io
from sqlalchemy import *

# start = datetime.now()

# table_name = 'SOCIAL_DATA'
# engine, sess = gbl.get_session(gbl.settings.CONN_STR)
# # engine.echo = True
# meta = MetaData()
# social_data = Table(table_name, meta, autoload=True, autoload_with=engine)
# for u in sess.query(social_data).filter(social_data.c.tags.ilike('%jpm%'))[0:10]:
#     if u.content is not None:
#         print(u.content.encode('utf-8') )


def get_table(table_name):
    engine, sess = gbl.get_session(gbl.settings.CONN_STR)
    # engine.echo = True
    meta = MetaData()
    return engine, sess, Table(table_name, meta, autoload=True, autoload_with=engine)
    #print(sess.query(social_data).filter(social_data.c.type.like('%Twitter%')).count())

# for u in sess.query(test.c.content)[0]:
#     print(u.encode('utf-8'))

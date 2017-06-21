from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import settings

def get_session(conn_str):
    engine = create_engine(conn_str, encoding='utf8', pool_timeout=90)
    Session = sessionmaker()
    Session.configure(bind=engine)

    return engine, Session()

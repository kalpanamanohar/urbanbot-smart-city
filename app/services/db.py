from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")

DEBUG = os.getenv("DEBUG", "False") == "True"

password = quote_plus(DB_PASSWORD)

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=280,
    echo=DEBUG
)

SessionLocal = sessionmaker(bind=engine)


def execute_query(sql, params=None):
    try:
        with engine.begin() as conn:
            result = conn.execute(text(sql), params or {})

            if result.returns_rows:
                return [dict(row._mapping) for row in result]

            return {"message": "Query executed successfully"}

    except Exception as e:
        return {"error": str(e)}

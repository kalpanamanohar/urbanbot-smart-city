"""
Analytics Engine
----------------
Executes read-only SQL queries safely and returns Pandas DataFrame.
Used by chatbot and analytics dashboard.
"""

import pandas as pd
from sqlalchemy import text
from services.db import engine
from utils.logger import get_logger

logger = get_logger()


def execute_sql(sql: str):
    """
    Execute a read-only SQL query and return DataFrame.
    """

    try:
        if not sql or not sql.strip():
            return "Invalid SQL query."

        # Prevent dangerous operations
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]

        if any(keyword in sql.upper() for keyword in forbidden_keywords):
            logger.warning("Blocked dangerous SQL attempt.")
            return "Only SELECT queries are allowed."

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        logger.info("SQL executed successfully.")
        return df

    except Exception as e:
        logger.error(f"Analytics SQL error: {e}")
        return "Database error occurred while executing query."

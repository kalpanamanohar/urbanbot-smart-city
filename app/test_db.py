from services.db import get_connection
from sqlalchemy import text

with get_connection() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())

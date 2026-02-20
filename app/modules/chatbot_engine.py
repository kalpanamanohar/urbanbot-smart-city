"""
Chatbot Engine
--------------
Main logic controller:
- Classifies intent
- Generates SQL
- Executes query
- Sends result to LLM
"""

from modules.intent_classifier import classify_intent
from modules.sql_generator import generate_sql
from modules.prompts import format_prompt
from modules.llm_client import ask_llm
from services.db import execute_query


def process_user_query(query: str) -> str:

    if not query:
        return "Please enter a valid question."

    # 1️⃣ Detect intent
    intent = classify_intent(query)

    # 2️⃣ General queries → direct LLM
    if intent == "general":
        return ask_llm(query)

    # 3️⃣ Generate SQL
    sql = generate_sql(query, intent)

    if not sql:
        return "Sorry, I couldn't understand your request."

    # 4️⃣ Execute SQL
    result = execute_query(sql)

    # 5️⃣ Handle database errors
    if isinstance(result, dict) and "error" in result:
        return f"Database Error: {result['error']}"

    # 6️⃣ No records
    if not result:
        return "No records found in the database."

    # 7️⃣ Convert list of dicts to readable string
    formatted_data = "\n".join(
        [", ".join(f"{key}: {value}" for key, value in row.items()) for row in result]
    )

    # 8️⃣ Format prompt for LLM explanation
    prompt = format_prompt(query, formatted_data)

    # 9️⃣ Ask LLM for explanation
    return ask_llm(prompt)

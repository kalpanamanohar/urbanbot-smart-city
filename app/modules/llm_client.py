"""
LLM Client Module
-----------------
Handles communication with Groq API.
Uses Mixtral model for explanations.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def ask_llm(prompt: str) -> str:
    """
    Sends prompt to Groq LLM and returns response.
    Includes error handling for production stability.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "LLM Error: GROQ_API_KEY is not set in environment variables."

    try:
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional Smart City analytics assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"

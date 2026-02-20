"""
Prompt Formatter Module
-----------------------
Formats database results into structured prompts
for the LLM to generate clear explanations.
"""

def format_prompt(question: str, result: str) -> str:
    """
    Creates a structured prompt for the LLM.

    Args:
        question (str): Original user question
        result (str): Database query result

    Returns:
        str: Formatted prompt
    """

    return f"""
You are a Smart City AI Analytics Assistant.

Your job:
- Explain database results clearly.
- Use simple and professional language.
- Do NOT use technical SQL terms.
- Do NOT guess or invent data.
- Base your explanation strictly on the provided result.

User Question:
{question}

Database Result:
{result}

Instructions:
1. Summarize the result in 2-4 clear sentences.
2. Highlight important insights.
3. If data shows a trend, mention it.
4. Keep explanation short and easy to understand.

Now provide the explanation:
"""

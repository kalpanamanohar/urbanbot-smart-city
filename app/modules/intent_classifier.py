"""
Intent Classifier Module
------------------------
Determines user intent based on keyword matching.
Used by chatbot_engine to route queries to correct SQL generator.
"""

def classify_intent(query: str) -> str:
    """
    Classify user query into predefined intents.

    Args:
        query (str): User input query

    Returns:
        str: Identified intent
    """

    if not query or not isinstance(query, str):
        return "general"

    q = query.lower().strip()

    # Intent keyword mapping
    intent_keywords = {
        "accident": [
            "accident", "crash", "collision", "road accident"
        ],
        "pothole": [
            "pothole", "road damage", "damaged road"
        ],
        "crowd": [
            "crowd", "people gathering", "public gathering", "overcrowded"
        ],
        "traffic": [
            "traffic", "vehicle", "vehicles", "congestion", "jam"
        ],
        "aqi": [
            "aqi", "pollution", "air quality", "air index"
        ],
        "complaint": [
            "complaint", "grievance", "issue", "problem report"
        ]
    }

    # Match intent
    for intent, keywords in intent_keywords.items():
        if any(keyword in q for keyword in keywords):
            return intent

    return "general"

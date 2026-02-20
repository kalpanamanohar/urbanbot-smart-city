"""
SQL Generator Module
--------------------
Generates safe SELECT SQL queries based on detected intent and user query.
Matches MySQL schema for UrbanBot database.
"""

def generate_sql(query: str, intent: str) -> str:
    q = query.lower()

    # ---------------- TIME FILTER ----------------
    time_filter = ""

    if "today" in q:
        time_filter = "WHERE DATE(event_time) = CURDATE()"

    elif "last 7 days" in q or "last week" in q:
        time_filter = "WHERE event_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"

    # ---------------- HELPER FLAGS ----------------
    is_highest = any(word in q for word in ["highest", "most", "top", "maximum"])
    is_lowest = any(word in q for word in ["lowest", "least", "minimum"])
    is_top3 = "top 3" in q

    # ---------------- ACCIDENT ----------------
    if intent == "accident":

        if is_top3:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM accident_events
                {time_filter}
                GROUP BY area
                ORDER BY total DESC
                LIMIT 3;
            """

        if is_highest:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM accident_events
                {time_filter}
                GROUP BY area
                ORDER BY total DESC
                LIMIT 1;
            """

        if is_lowest:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM accident_events
                {time_filter}
                GROUP BY area
                ORDER BY total ASC
                LIMIT 1;
            """

        return f"""
            SELECT COUNT(*) AS total
            FROM accident_events
            {time_filter};
        """

    # ---------------- POTHOLE ----------------
    if intent == "pothole":

        if is_highest:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM pothole_events
                {time_filter}
                GROUP BY area
                ORDER BY total DESC
                LIMIT 1;
            """

        return f"""
            SELECT COUNT(*) AS total
            FROM pothole_events
            {time_filter};
        """

    # ---------------- CROWD ----------------
    if intent == "crowd":

        if is_highest:
            return f"""
                SELECT area, AVG(crowd_count) AS avg_crowd
                FROM crowd_events
                {time_filter}
                GROUP BY area
                ORDER BY avg_crowd DESC
                LIMIT 1;
            """

        if is_lowest:
            return f"""
                SELECT area, AVG(crowd_count) AS avg_crowd
                FROM crowd_events
                {time_filter}
                GROUP BY area
                ORDER BY avg_crowd ASC
                LIMIT 1;
            """

        return f"""
            SELECT AVG(crowd_count) AS avg_crowd
            FROM crowd_events
            {time_filter};
        """

    # ---------------- TRAFFIC ----------------
    if intent == "traffic":

        if is_top3:
            return f"""
                SELECT area, AVG(predicted_vehicles) AS avg_traffic
                FROM traffic_events
                {time_filter}
                GROUP BY area
                ORDER BY avg_traffic DESC
                LIMIT 3;
            """

        if is_highest:
            return f"""
                SELECT area, AVG(predicted_vehicles) AS avg_traffic
                FROM traffic_events
                {time_filter}
                GROUP BY area
                ORDER BY avg_traffic DESC
                LIMIT 1;
            """

        if is_lowest:
            return f"""
                SELECT area, AVG(predicted_vehicles) AS avg_traffic
                FROM traffic_events
                {time_filter}
                GROUP BY area
                ORDER BY avg_traffic ASC
                LIMIT 1;
            """

        return f"""
            SELECT AVG(predicted_vehicles) AS avg_traffic
            FROM traffic_events
            {time_filter};
        """

    # ---------------- AQI ----------------
    if intent == "aqi":

        if is_highest or "worst" in q:
            return f"""
                SELECT city, AVG(predicted_aqi) AS avg_aqi
                FROM aqi_events
                {time_filter}
                GROUP BY city
                ORDER BY avg_aqi DESC
                LIMIT 1;
            """

        if is_lowest or "best" in q:
            return f"""
                SELECT city, AVG(predicted_aqi) AS avg_aqi
                FROM aqi_events
                {time_filter}
                GROUP BY city
                ORDER BY avg_aqi ASC
                LIMIT 1;
            """

        return f"""
            SELECT AVG(predicted_aqi) AS avg_aqi
            FROM aqi_events
            {time_filter};
        """

    # ---------------- COMPLAINT ----------------
    if intent == "complaint":

        if is_highest:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM citizen_complaints
                {time_filter}
                GROUP BY area
                ORDER BY total DESC
                LIMIT 1;
            """

        if is_lowest:
            return f"""
                SELECT area, COUNT(*) AS total
                FROM citizen_complaints
                {time_filter}
                GROUP BY area
                ORDER BY total ASC
                LIMIT 1;
            """

        return f"""
            SELECT COUNT(*) AS total
            FROM citizen_complaints
            {time_filter};
        """

    return None

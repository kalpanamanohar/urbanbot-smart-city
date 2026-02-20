# services/email_service.py

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
logger = get_logger()


def send_email_alert(module, subject, body):

    try:
        sender_email = os.getenv("EMAIL_ADDRESS")
        password = os.getenv("EMAIL_PASSWORD")

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = sender_email
        message["Subject"] = f"[UrbanBot - {module}] {subject}"

        message.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(message)
        server.quit()

        logger.info("Email alert sent successfully")

    except Exception as e:
        logger.error(f"Email sending failed: {e}")

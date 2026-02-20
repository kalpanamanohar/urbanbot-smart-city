import mysql.connector

conn = mysql.connector.connect(
    host="database-1.c5e4oogg6srw.ap-south-1.rds.amazonaws.com",
    
    user="admin",
    password="Urbanbot#2026",
    database="urbanbot",
    port = 3306
)

print("Connected Successfully")

import mysql.connector
from AllinOne import data, responses, severity, Urls
from datetime import datetime
import os

db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
)

cur = db.cursor()

# 1. Delete news older than 24 hours
cur.execute("""
    DELETE FROM news
    WHERE time < NOW() - INTERVAL 24 HOUR
""")

# 2. Current exact timestamp
time = datetime.now()

# 3. Insert new news
query = """
INSERT INTO news
(short, headline, severity, time, source)
VALUES (%s, %s, %s, %s, %s)
"""

for i, j, k, u in zip(responses, data, severity, Urls):
    cur.execute(query, (i, j, k, time, u))

db.commit()

print("database execution done")

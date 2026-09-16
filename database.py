import mysql.connector
from AllinOne import data, responses, severity, Urls, links
from datetime import datetime
import os

new_data = []

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

# 2. Get headlines already present in database
cur.execute("SELECT headline FROM news")
old_data = {row[0] for row in cur.fetchall()}

# 3. Keep only new headlines
for headline in data:
    if headline not in old_data:
        new_data.append(headline)

# 4. Current exact timestamp
time = datetime.now()

# 5. Insert only new news
query = """
INSERT INTO news
(short, headline, severity, time, source, link)
VALUES (%s, %s, %s, %s, %s, %s)
"""

# IMPORTANT:
# responses, severity and Urls must correspond to data.
# Therefore we need their indexes to select the matching values.

for index, headline in enumerate(data):
    if headline not in old_data:

        cur.execute(
            query,
            (
                responses[index],
                headline,
                severity[index],
                time,
                Urls[index]
                links[index]
            )
        )

db.commit()

print("database execution done")

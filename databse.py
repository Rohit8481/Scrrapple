import mysql.connector
from AllinOne import data, responses, severity
from datetime import datetime 


############ FOR TIME  ################
time = str(datetime.now().hour)
#########################################

db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
    port=int(os.getenv("TIDB_PORT", "4000"))
)
    
cur = db.cursor()
query = f"INSERT INTO news (id, short, headline, severity, time) VALUES (%s, %s, %s, %s, %s);"
cur.execute("TRUNCATE TABLE news;")
id = 101

for  i, j, k in zip( responses, data, severity ):
    cur.execute(query, ( id, i, j, k, time))
    id += 1 
db.commit()
print("database execution done")

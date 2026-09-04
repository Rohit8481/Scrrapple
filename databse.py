import mysql.connector
from AllinOne import data, responses, severity
from datetime import datetime 


############ FOR TIME  ################
time = str(datetime.now().hour)
#########################################

db = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    user="2MeFkK3RJd7rXvv.root",
    password="N2UuuYUcJmeZUsry",
    database="newz_app"
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
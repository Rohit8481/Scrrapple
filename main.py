from fastapi import FastAPI
import mysql.connector

app = FastAPI()


@app.get("/")
def news():
  db = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    user="2MeFkK3RJd7rXvv.root",
    password="N2UuuYUcJmeZUsry",
    database="newz_app"
)

  cur = db.cursor(dictionary=True)  # dictionary=True se clean JSON milta h
  cur.execute("SELECT * FROM news;")
  results = cur.fetchall()
  db.close()  
  return results
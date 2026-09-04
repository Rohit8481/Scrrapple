from fastapi import FastAPI
import mysql.connector

app = FastAPI()


@app.get("/")
def news():
  db = mysql.connector.connect(
    host=os.getenv("TIDB_HOST"),
    user=os.getenv("TIDB_USER"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE"),
    port=int(os.getenv("TIDB_PORT", "4000"))
)

  cur = db.cursor(dictionary=True)  # dictionary=True se clean JSON milta h
  cur.execute("SELECT * FROM news;")
  results = cur.fetchall()
  db.close()  
  return results

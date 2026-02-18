import psycopg

DB_URI = "postgresql://postgres:prem123@localhost:5432/langgraph_db"

try:
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("✅ Connection successful!")
except Exception as e:
    print("❌ Error:", e)


conn = psycopg.connect(
    host="localhost",
    port="5432",
    user="postgres",
    password="prem123",
    dbname="langgraph_db"
)

cur = conn.cursor()

cur.execute("""
SELECT state
FROM langgraph_checkpoint
WHERE thread_id = 'user_1'
ORDER BY checkpoint_id DESC
LIMIT 1;
""")

row = cur.fetchone()

if row:
    state = row[0]
    messages = state["channel_values"]["messages"]
    
    print("Conversation for user_1:")
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content']}")

cur.close()
conn.close()
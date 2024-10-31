import os
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TZ_INFO = os.getenv("TZ", "Europe/Berlin")
tz = ZoneInfo(TZ_INFO)

def get_db_connection():
    """Create a connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST_LOCAL"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        with conn.cursor() as cur:
            # Create conversations table with session_id
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            
            # Create feedback table with comment field and timestamp
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    comment TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    CONSTRAINT fk_conversation
                        FOREIGN KEY(conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)
        conn.commit()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        conn.rollback()
    finally:
        conn.close()
        
def save_conversation(conversation_id, question, answer_data, timestamp=None):
    """Save a conversation to the database."""
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to save conversation due to connection failure.")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp
                ),
            )
        conn.commit()
        logging.info(f"Conversation saved successfully. ID: {conversation_id}")
    except Exception as e:
        logging.error(f"Failed to save conversation: {e}")
        conn.rollback()
    finally:
        conn.close()

def save_feedback(conversation_id, feedback, timestamp=None):
    """Save user feedback for a conversation."""
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to save feedback due to connection failure.")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
        logging.info(f"Feedback saved successfully for conversation ID: {conversation_id}")
    except Exception as e:
        logging.error(f"Failed to save feedback: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_recent_conversations(limit=5, relevance=None):
    """Retrieve recent conversations with optional relevance filter."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to get recent conversations due to connection failure.")
        return []
    
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            if relevance:
                query += " WHERE c.relevance = %s"
            query += " ORDER BY c.timestamp DESC LIMIT %s"

            params = (relevance, limit) if relevance else (limit,)
            cur.execute(query, params)
            return cur.fetchall()
    except Exception as e:
        logging.error(f"Failed to fetch recent conversations: {e}")
        return []
    finally:
        conn.close()

def get_relevance_stats():
    """Get statistics about conversation relevance."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to get relevance stats due to connection failure.")
        return []
    
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    relevance,
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
                FROM conversations
                GROUP BY relevance
            """)
            return cur.fetchall()
    except Exception as e:
        logging.error(f"Failed to fetch relevance stats: {e}")
        return []
    finally:
        conn.close()

def get_conversation_by_id(conversation_id):
    """Retrieve a specific conversation by ID."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to get conversation due to connection failure.")
        return None
    
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
                WHERE c.id = %s
            """, (conversation_id,))
            return cur.fetchone()
    except Exception as e:
        logging.error(f"Failed to fetch conversation: {e}")
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
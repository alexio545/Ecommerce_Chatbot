import os
import time
import uuid
import logging
import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager

from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from groq import Groq
from redis import Redis
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Conversation:
    id: str
    session_id: str
    question: str
    answer: str
    model_used: str
    response_time: float
    relevance: str
    relevance_explanation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    eval_prompt_tokens: int
    eval_completion_tokens: int
    eval_total_tokens: int
    timestamp: datetime

@dataclass
class Feedback:
    id: Optional[str]
    conversation_id: str
    feedback: int
    comment: str
    timestamp: datetime


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class DatabaseManager:
    """Manages all database operations with proper connection handling."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def save_conversation(self, conversation: Conversation) -> bool:
        """Stores a conversation in the database."""
        query = """
            INSERT INTO conversations (
                id, session_id, question, answer, model_used, 
                response_time, relevance, relevance_explanation,
                prompt_tokens, completion_tokens, total_tokens,
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens,
                openai_cost, timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        
        # Calculate OpenAI cost based on token usage
        openai_cost = calculate_groq_cost({
            'prompt_tokens': conversation.prompt_tokens + conversation.eval_prompt_tokens,
            'completion_tokens': conversation.completion_tokens + conversation.eval_completion_tokens
        })
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    conversation.id,
                    conversation.session_id,
                    conversation.question,
                    conversation.answer,
                    conversation.model_used,
                    conversation.response_time,
                    conversation.relevance,
                    conversation.relevance_explanation,
                    conversation.prompt_tokens,
                    conversation.completion_tokens,
                    conversation.total_tokens,
                    conversation.eval_prompt_tokens,
                    conversation.eval_completion_tokens,
                    conversation.eval_total_tokens,
                    openai_cost,
                    conversation.timestamp
                ))
                conn.commit()
                return True






class DatabaseManager:
    """Manages all database operations with proper connection handling."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def save_conversation(self, conversation: Conversation) -> bool:
        """Stores a conversation in the database."""
        query = """
            INSERT INTO conversations (
                id, session_id, question, answer, model_used, 
                response_time, relevance, relevance_explanation,
                prompt_tokens, completion_tokens, total_tokens,
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens,
                timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    conversation.id, conversation.session_id,
                    conversation.question, conversation.answer,
                    conversation.model_used, conversation.response_time,
                    conversation.relevance, conversation.relevance_explanation,
                    conversation.prompt_tokens, conversation.completion_tokens,
                    conversation.total_tokens, conversation.eval_prompt_tokens,
                    conversation.eval_completion_tokens, conversation.eval_total_tokens,
                    conversation.timestamp
                ))
                conn.commit()
                return True

    def get_feedback_stats(self) -> Dict[str, int]:
        """Retrieves overall feedback statistics."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT 
                        SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as thumbs_up,
                        SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END) as thumbs_down
                    FROM feedback
                """)
                result = cur.fetchone()
                return {
                    "thumbs_up": result["thumbs_up"] or 0,
                    "thumbs_down": result["thumbs_down"] or 0
                }

class ImprovedFeedbackManager:
    """Enhanced feedback management with better error handling and validation."""
    
    def __init__(self, db_manager: DatabaseManager, tz: ZoneInfo):
        self.db_manager = db_manager
        self.tz = tz

    def validate_feedback(self, feedback: int, conversation_id: str) -> None:
        """Validates feedback data."""
        if feedback not in (0, 1):
            raise ValueError("Feedback must be either 0 or 1")
        if not conversation_id:
            raise ValueError("Conversation ID is required")

    def add_feedback(self, conversation_id: str, feedback: int, comment: str = "") -> bool:
        """Stores feedback with improved validation and error handling."""
        try:
            self.validate_feedback(feedback, conversation_id)
            feedback_obj = Feedback(
                id=None,
                conversation_id=conversation_id,
                feedback=feedback,
                comment=comment,
                timestamp=datetime.now(self.tz)
            )
            return self._store_feedback(feedback_obj)
        except ValueError as e:
            logging.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error storing feedback: {str(e)}")
            return False



    
    def _store_feedback(self, feedback: Feedback) -> bool:
        """Internal method to store feedback in the database."""
        query = """
            INSERT INTO feedback (conversation_id, feedback, comment, timestamp)
            VALUES (%s, %s, %s, %s)
            """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Verify the conversation exists first
                cur.execute(
                    "SELECT id FROM conversations WHERE id = %s",
                    (feedback.conversation_id,)
                )
                if cur.fetchone() is None:
                    # If conversation doesn't exist, we need to save it first
                    # Generate a session_id if not provided
                    session_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO conversations 
                        (id, session_id, question, answer, model_used, 
                        response_time, relevance, relevance_explanation, 
                        prompt_tokens, completion_tokens, total_tokens,
                        eval_prompt_tokens, eval_completion_tokens, 
                        eval_total_tokens, openai_cost, timestamp) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            feedback.conversation_id, 
                            session_id,  # session_id
                            "Unknown Question", 
                            "No Answer Recorded", 
                            "unknown",  # model_used
                            0.0,        # response_time
                            "UNKNOWN",  # relevance
                            "",         # relevance_explanation
                            0,          # prompt_tokens
                            0,          # completion_tokens
                            0,          # total_tokens
                            0,          # eval_prompt_tokens
                            0,          # eval_completion_tokens
                            0,          # eval_total_tokens
                            0.0,        # openai_cost
                            feedback.timestamp
                        )
                    )
                
                # Now insert the feedback
                cur.execute(query, (
                    feedback.conversation_id,
                    feedback.feedback,
                    feedback.comment,
                    feedback.timestamp
                ))
                conn.commit()
                return True

    def get_feedback_stats(self) -> Dict[str, int]:
        """Get feedback statistics."""
        return self.db_manager.get_feedback_stats()

class ImprovedRedisMemoryManager:
    """Enhanced Redis memory management with better serialization and error handling."""
    
    def __init__(self, redis_url: str, db_manager: DatabaseManager, 
                 expiration_time: int = 3600):
        self.redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.session_id = str(uuid.uuid4())
        self.expiration_time = expiration_time
        self.db_manager = db_manager
        self.feedback_manager = ImprovedFeedbackManager(
            db_manager=db_manager,
            tz=ZoneInfo(os.getenv("TZ", "UTC"))
        )

    def _get_key(self) -> str:
        return f"conversation:{self.session_id}"

    def serialize_message(self, message: Dict[str, Any]) -> str:
        """Serializes message with error handling."""
        try:
            return json.dumps(message)
        except json.JSONEncodeError as e:
            logging.error(f"Error serializing message: {str(e)}")
            raise

    def deserialize_message(self, message: str) -> Dict[str, Any]:
        """Deserializes message with error handling."""
        try:
            return json.loads(message)
        except json.JSONDecodeError as e:
            logging.error(f"Error deserializing message: {str(e)}")
            raise

    def add_message(self, role: str, content: str, metadata: Dict[str, Any]) -> None:
        """Adds a message to Redis and optionally persists to PostgreSQL."""
        key = self._get_key()
        message = {
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Redis
        serialized_message = self.serialize_message(message)
        self.redis_client.rpush(key, serialized_message)
        self.redis_client.expire(key, self.expiration_time)
        
        # If it's a complete conversation (both question and answer), persist to PostgreSQL
        if role == "assistant" and metadata.get("conversation_id"):
            self._persist_conversation(message, metadata)

    def _get_last_user_message(self) -> Dict[str, Any]:
        """Retrieves the last user message from the conversation history."""
        messages = self.get_conversation_history()
        for message in reversed(messages):
            if message["role"] == "user":
                return message
        raise ValueError("No user message found in conversation history")

    def _persist_conversation(self, message: Dict[str, Any], 
                            metadata: Dict[str, Any]) -> None:
        """Persists the conversation to PostgreSQL."""
        try:
            last_user_message = self._get_last_user_message()
            conversation = Conversation(
                id=metadata["conversation_id"],
                session_id=self.session_id,
                question=last_user_message["content"],
                answer=message["content"],
                model_used=metadata.get("model_used", "unknown"),
                response_time=metadata.get("response_time", 0.0),
                relevance=metadata.get("relevance", "unknown"),
                relevance_explanation=metadata.get("relevance_explanation", ""),
                prompt_tokens=metadata.get("prompt_tokens", 0),
                completion_tokens=metadata.get("completion_tokens", 0),
                total_tokens=metadata.get("total_tokens", 0),
                eval_prompt_tokens=metadata.get("eval_prompt_tokens", 0),
                eval_completion_tokens=metadata.get("eval_completion_tokens", 0),
                eval_total_tokens=metadata.get("eval_total_tokens", 0),
                timestamp=datetime.now(ZoneInfo("UTC"))
            )
            
            self.db_manager.save_conversation(conversation)
        except Exception as e:
            logging.error(f"Error persisting conversation: {str(e)}")
            # Continue execution even if persistence fails

    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves conversation history from Redis."""
        key = self._get_key()
        messages = self.redis_client.lrange(key, -limit, -1)
        return [self.deserialize_message(msg) for msg in messages]

    def clear_history(self) -> None:
        """Clears the conversation history from Redis."""
        key = self._get_key()
        self.redis_client.delete(key)

class RAGSystem:
    def __init__(self, 
                 es_url: str = 'http://localhost:9200',
                 model_name: str = 'multi-qa-MiniLM-L6-cos-v1',
                 redis_url: str = 'redis://localhost:6379',
                 memory_expiration: int = 3600):
        self.es_client = Elasticsearch([es_url])
        self.model = SentenceTransformer(model_name)
        self.db_manager = DatabaseManager(database_url=f"host={os.getenv('POSTGRES_HOST_LOCAL')} " \
                 f"dbname={os.getenv('POSTGRES_DB')} " \
                 f"user={os.getenv('POSTGRES_USER')} " \
                 f"password={os.getenv('POSTGRES_PASSWORD')} " \
                 f"port={os.getenv('POSTGRES_PORT')}")
        self.memory = ImprovedRedisMemoryManager(
            redis_url=redis_url,
            db_manager=self.db_manager,
            expiration_time=memory_expiration
        )
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.session_id = self.memory.session_id

    def elastic_search_hybrid(self, query: str, index_name: str = "ecommerce-products") -> list:
        """Performs hybrid search combining vector similarity and keyword matching."""
        query_vector = self.model.encode(query).tolist()
        
        search_query = {
            'size': 5,
            'query': {
                'bool': {
                    'must': [
                        {
                            'script_score': {
                                'query': {'match_all': {}},
                                'script': {
                                    'source': "cosineSimilarity(params.query_vector, 'combined_vector') + 1.0",
                                    'params': {'query_vector': query_vector}
                                }
                            }
                        }
                    ],
                    'should': [
                        {
                            'multi_match': {
                                'query': query,
                                'fields': ['productName^3', 'category^2', 'productDescription'],
                                'type': 'cross_fields',
                                'operator': 'and'
                            }
                        }
                    ]
                }
            }
        }
        
        try:
            results = self.es_client.search(index=index_name, body=search_query)
            return [hit['_source'] for hit in results['hits']['hits']]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []
            
    def get_feedback(self, conversation_id: str) -> str:
        """Get user feedback for a response."""
        while True:
            try:
                print("\nWas this response helpful?")
                print("1: 👍 Thumbs up")
                print("0: 👎 Thumbs down")
                feedback = input("Your feedback (1 or 0): ").strip()
                
                if feedback not in ['0', '1']:
                    print("Please enter either 1 for thumbs up or 0 for thumbs down.")
                    continue
                
                comment = input("Additional comments (optional): ").strip()
                
                success = self.memory.feedback_manager.add_feedback(
                    conversation_id=conversation_id,
                    feedback=int(feedback),
                    comment=comment
                )
                
                if success:
                    stats = self.memory.feedback_manager.get_feedback_stats()
                    return f"Thank you for your feedback! Current stats: 👍 {stats['thumbs_up']} | 👎 {stats['thumbs_down']}"
                else:
                    return "Thank you for your feedback! (Note: There was an issue storing the feedback)"
                
            except Exception as e:
                logging.error(f"Error getting feedback: {str(e)}")
                return "Sorry, there was an error processing your feedback."

    def build_prompt(self, query: str, search_results: list, conversation_history: List[Dict[str, Any]]) -> str:
        """Builds a prompt including conversation history and product information."""
        context_items = []
        for product in search_results:
            final_price = product['price'] * (1 - product.get('discount', 0)/100)
            context_items.append(
                f"Product: {product['productName']}\n"
                f"Category: {product['category']}\n"
                f"Price: ${product['price']:.2f}\n"
                f"Final Price: ${final_price:.2f}\n"
                f"Colors: {', '.join(product['availableColours'])}\n"
                f"Sizes: {', '.join(product['sizes'])}\n"
                f"Description: {product['productDescription']}\n"
            )
        
        context = "\n---\n".join(context_items)
        history_lines = []
        for msg in conversation_history:
            history_lines.append(f"{msg['role']}: {msg['content']}")
        history = "\n".join(history_lines)
        
        return f"""You are a knowledgeable and helpful e-commerce shopping assistant. Using the provided product information 
        and conversation history, answer the customer's question accurately and professionally.

        Previous Conversation:
        {history}

        Product Information:
        {context}

        Current Question: {query}

        Please provide a helpful, accurate, and natural response that takes into account both the conversation history and the 
        available product information."""

    def generate_response(self, prompt: str, model: str = 'llama-3.1-70b-versatile') -> Tuple[str, Dict[str, Any], float]:
        """Generates a response using the Groq LLM."""
        start_time = time.time()
        response = self.groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        
        return response.choices[0].message.content, response.usage.to_dict(), end_time - start_time

    def process_query(self, query: str, model: str = 'llama-3.1-70b-versatile') -> Dict[str, Any]:
        """Process a user query and return the response with metadata."""
        conversation_id = str(uuid.uuid4())
        search_results = self.elastic_search_hybrid(query)
        
        metadata = {
            'conversation_id': conversation_id,
            'search_count': len(search_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.memory.add_message("user", query, metadata)
        
        if not search_results:
            response = "I apologize, but I couldn't find any relevant products matching your query. Could you please try rephrasing your question?"
            self.memory.add_message("assistant", response, metadata)
            return {
                   "id": conversation_id,
                    "session_id": self.session_id,
                    "question": query,
                    "answer": response,
                    "model_used": model,
                    "response_time": response_time,
                    "relevance": relevance,
                    "relevance_explanation": explanation,
                    "prompt_tokens": usage_stats["prompt_tokens"],
                    "completion_tokens": usage_stats["completion_tokens"],
                    "total_tokens": usage_stats["total_tokens"],
                    "eval_prompt_tokens": eval_usage["prompt_tokens"],
                    "eval_completion_tokens": eval_usage["completion_tokens"],
                    "eval_total_tokens": eval_usage["total_tokens"],
                    "timestamp": datetime.now(ZoneInfo("UTC"))
                            }
        
        conversation_history = self.memory.get_conversation_history()
        prompt = self.build_prompt(query, search_results, conversation_history)
        
        response, usage_stats, response_time = self.generate_response(prompt, model)
        
        # Evaluate the response using LLM
        relevance, explanation, eval_usage = evaluate_relevance(query, response, self.groq_client, model)
        
        self.memory.add_message("assistant", response, metadata)
        
        return {
           "id": conversation_id,
            "session_id": self.session_id,
            "question": query,
            "answer": response,
            "model_used": model,
            "response_time": response_time,
            "relevance": relevance,
            "relevance_explanation": explanation,
            "prompt_tokens": usage_stats["prompt_tokens"],
            "completion_tokens": usage_stats["completion_tokens"],
            "total_tokens": usage_stats["total_tokens"],
            "eval_prompt_tokens": eval_usage["prompt_tokens"],
            "eval_completion_tokens": eval_usage["completion_tokens"],
            "eval_total_tokens": eval_usage["total_tokens"],
            "timestamp": datetime.now(ZoneInfo("UTC"))
                }

def evaluate_relevance(question: str, answer: str, groq_client, model_choice: str = 'llama-3.1-70b-versatile') -> Tuple[str, str, Dict[str, Any]]:
    """Evaluates the relevance of an answer using the Groq LLM."""
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON format:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)

    start_time = time.time()
    response = groq_client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.time()
    
    evaluation_response = response.choices[0].message.content
    usage_stats = response.usage.to_dict()

    try:
        json_eval = json.loads(evaluation_response)
        return json_eval['Relevance'], json_eval['Explanation'], usage_stats
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", usage_stats

def calculate_groq_cost(tokens: Dict[str, Any], pricing: Dict[str, float] = None) -> float:
    """Calculates cost based on token usage for Groq LLM."""
    if pricing is None:
        pricing = {
            'prompt_token_cost': 0.02,
            'completion_token_cost': 0.05
        }

    cost = (tokens.get('prompt_tokens', 0) * pricing['prompt_token_cost'] +
            tokens.get('completion_tokens', 0) * pricing['completion_token_cost']) / 1000

    return cost

def main():
    parser = argparse.ArgumentParser(description='E-commerce Product Assistant')
    parser.add_argument('--model', type=str, default='llama-3.1-70b-versatile',
                       help='Model to use for generating responses')
    parser.add_argument('--elastic-url', type=str, default='http://localhost:9200',
                       help='Elasticsearch URL')
    parser.add_argument('--redis-url', type=str, default='redis://localhost:6379',
                       help='Redis URL')
    
    args = parser.parse_args()
    
    rag = RAGSystem(
        es_url=args.elastic_url,
        model_name='multi-qa-MiniLM-L6-cos-v1',
        redis_url=args.redis_url,
        memory_expiration=3600
    )

    print("E-commerce Product Assistant")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if query.lower() == 'clear':
            rag.memory.clear_history()
            print("Conversation history cleared!")
            continue
        
        if not query:
            continue
        
        try:
            result = rag.process_query(query, model=args.model)
            print("\nAssistant:", result['answer'])
            
            print("\n" + "="*20 + " Response Evaluation " + "="*20)
            print(f"Relevance: {result['relevance']}")
            print(f"Explanation: {result['relevance_explanation']}")
            
            print("\n" + "="*20 + " Response Metrics " + "="*20)
            print(f"Response time: {result['response_time']:.2f} seconds")
            print(f"Response tokens: {result['total_tokens']}")
            print(f"Evaluation tokens: {result['eval_total_tokens']}")
            
            total_cost = calculate_groq_cost({
                'prompt_tokens': result['prompt_tokens'] + result['eval_prompt_tokens'],
                'completion_tokens': result['completion_tokens'] + result['eval_completion_tokens']
            })
            print(f"Total estimated cost: ${total_cost:.4f}")
            
            feedback_result = rag.get_feedback(result['id'])
            print("\n" + "="*20 + " User Feedback " + "="*20)
            print(feedback_result)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
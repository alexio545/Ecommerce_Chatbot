import os
import time
import uuid
import logging
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Tuple, List, Optional
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from groq import Groq
from redis import Redis
from dotenv import load_dotenv
import json
import psycopg2
from psycopg2.extras import DictCursor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Create a database connection."""
    try:
        return psycopg2.connect(os.getenv('DATABASE_URL'))
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return None

class FeedbackManager:
    """Manages user feedback in PostgreSQL."""
    
    def __init__(self):
        """Initialize FeedbackManager."""
        self.tz = ZoneInfo(os.getenv("TZ", "UTC"))
    
    def add_feedback(self, conversation_id: str, feedback: int, comment: str = "") -> bool:
        """Stores feedback for a specific conversation."""
        conn = get_db_connection()
        if conn is None:
            return False
        
        try:
            timestamp = datetime.now(self.tz)
            with conn.cursor() as cur:
                # First verify the conversation exists
                cur.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
                if cur.fetchone() is None:
                    logging.error(f"Conversation {conversation_id} not found")
                    return False
                
                # Insert the feedback
                cur.execute("""
                    INSERT INTO feedback (conversation_id, feedback, comment, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (conversation_id, feedback, comment, timestamp))
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Error storing feedback: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_feedback_stats(self) -> Dict[str, int]:
        """Retrieves overall feedback statistics."""
        conn = get_db_connection()
        if conn is None:
            return {"thumbs_up": 0, "thumbs_down": 0}
        
        try:
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
        except Exception as e:
            logging.error(f"Error getting feedback stats: {e}")
            return {"thumbs_up": 0, "thumbs_down": 0}
        finally:
            conn.close()

class RedisMemoryManager:
    """Manages conversation history in Redis with automatic expiration."""
    
    def __init__(self, redis_url='redis://localhost:6379', expiration_time=3600):
        self.redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.session_id = str(uuid.uuid4())
        self.expiration_time = expiration_time
        self.feedback_manager = FeedbackManager()
    
    def _get_key(self) -> str:
        return f"conversation:{self.session_id}"
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any]) -> None:
        key = self._get_key()
        message = {
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, self.expiration_time)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        key = self._get_key()
        messages = self.redis_client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    
    def clear_history(self) -> None:
        key = self._get_key()
        self.redis_client.delete(key)

class RAGSystem:
    def __init__(self, 
                 es_url='http://localhost:9200',
                 model_name='multi-qa-MiniLM-L6-cos-v1',
                 memory_expiration=3600):
        self.es_client = Elasticsearch([es_url])
        self.model = SentenceTransformer(model_name)
        self.memory = RedisMemoryManager(expiration_time=memory_expiration)
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
                "response_time": 0,
                "relevance": "no_results",
                "relevance_explanation": "No matching products found in the database.",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "eval_prompt_tokens": 0,
                "eval_completion_tokens": 0,
                "eval_total_tokens": 0,
                "openai_cost": 0,
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
        memory_expiration=3600
    )
    print("E-commerce Product Assistant")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("-" * 50)
    
    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        
        # Check for exit commands
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Check for clear history command
        if query.lower() == 'clear':
            rag.memory.clear_history()
            print("Conversation history cleared!")
            continue
        
        # Skip empty queries
        if not query:
            continue
        
        try:
            # Process the query
            result = rag.process_query(query, model=args.model)
            
            # Print the response
            print("\nAssistant:", result['answer'])
            
            # Print evaluation results
            print("\n" + "="*20 + " Response Evaluation " + "="*20)
            print(f"Relevance: {result['relevance']}")
            print(f"Explanation: {result['relevance_explanation']}")
            
            # Print detailed metadata
            print("\n" + "="*20 + " Response Metrics " + "="*20)
            print(f"Response time: {result['response_time']:.2f} seconds")
            print(f"Response tokens: {result['total_tokens']}")
            print(f"Evaluation tokens: {result['eval_total_tokens']}")
            total_cost = calculate_groq_cost({
                'prompt_tokens': result['prompt_tokens'] + result['eval_prompt_tokens'],
                'completion_tokens': result['completion_tokens'] + result['eval_completion_tokens']
            })
            print(f"Total estimated cost: ${total_cost:.4f}")
            
            # Get feedback
            feedback_result = rag.get_feedback(result['id'])
            print("\n" + "="*20 + " User Feedback " + "="*20)
            print(feedback_result)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
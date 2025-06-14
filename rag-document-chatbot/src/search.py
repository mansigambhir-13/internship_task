"""
Advanced Semantic Search & Answer Generation Script
Performs semantic search using Qdrant vector database and generates answers
Uses cosine similarity to find most relevant chunks and generates comprehensive answers
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("[OK] Loaded .env file")
            return True
    except ImportError:
        pass
    
    env_files = [Path('.env'), project_root / '.env']
    for env_file in env_files:
        if env_file.exists():
            print(f"[INFO] Loading .env from: {env_file}")
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"').strip("'")
                print(f"[OK] Loaded environment variables")
                return True
            except Exception as e:
                print(f"[ERROR] Error reading {env_file}: {e}")
    return False

load_env_file()

@dataclass
class SearchConfig:
    """Configuration for semantic search"""
    
    # Qdrant settings
    COLLECTION_NAME: str = "document_chunks"
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', '6333'))
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Search settings
    TOP_K_CHUNKS: int = int(os.getenv('TOP_K_CHUNKS', '20'))  # Initial retrieval
    FINAL_CHUNKS: int = int(os.getenv('FINAL_CHUNKS', '3'))   # Final selection for answer
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    
    # Answer generation settings
    ANSWER_MODEL: str = os.getenv('ANSWER_MODEL', 'gpt-3.5-turbo')
    MAX_CONTEXT_LENGTH: int = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration"""
        if self.QDRANT_URL and self.QDRANT_API_KEY:
            return {"url": self.QDRANT_URL, "api_key": self.QDRANT_API_KEY}
        else:
            return {"host": self.QDRANT_HOST, "port": self.QDRANT_PORT}

# Initialize configuration
config = SearchConfig()

# Setup logging
def setup_logging() -> logging.Logger:
    """Setup logging for search operations"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "logs" / "search.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SemanticSearchEngine:
    """Advanced semantic search engine with answer generation"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.qdrant_client = None
        self.embedding_client = None
        self.answer_client = None
        self.embedding_service = None
        self.embedding_dim = None
        
        # Initialize services
        self._initialize_qdrant_client()
        self._initialize_embedding_service()
        self._initialize_answer_generation()
        
        logger.info("SemanticSearchEngine initialized successfully")
    
    def _initialize_qdrant_client(self):
        """Initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient
            
            qdrant_config = self.config.get_qdrant_config()
            
            if "url" in qdrant_config and qdrant_config["url"]:
                self.qdrant_client = QdrantClient(
                    url=qdrant_config["url"],
                    api_key=qdrant_config["api_key"],
                    timeout=60.0
                )
                logger.info(f"[OK] Connected to Qdrant Cloud")
            else:
                self.qdrant_client = QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config["port"],
                    timeout=30.0
                )
                logger.info(f"[OK] Connected to local Qdrant")
            
            # Verify collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.COLLECTION_NAME not in collection_names:
                raise ValueError(f"Collection '{self.config.COLLECTION_NAME}' not found. Available: {collection_names}")
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.config.COLLECTION_NAME)
            self.embedding_dim = collection_info.config.params.vectors.size
            
            logger.info(f"[OK] Collection '{self.config.COLLECTION_NAME}' ready ({collection_info.points_count:,} points)")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Qdrant: {e}")
            raise
    
    def _initialize_embedding_service(self):
        """Initialize embedding service"""
        try:
            # Try OpenAI first
            if (self.config.EMBEDDING_MODEL.startswith("text-embedding") and 
                self.config.OPENAI_API_KEY):
                
                import openai
                self.embedding_client = openai.OpenAI(
                    api_key=self.config.OPENAI_API_KEY,
                    timeout=60.0
                )
                self.embedding_service = "openai"
                
                # Test connection
                test_response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input="test query"
                )
                logger.info(f"[OK] OpenAI embedding service ready: {self.config.EMBEDDING_MODEL}")
            
            # Try Gemini if OpenAI not available
            elif self.config.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                self.embedding_client = genai
                self.embedding_service = "gemini"
                self.config.EMBEDDING_MODEL = "text-embedding-004"
                
                # Test connection
                test_response = genai.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content="test query",
                    task_type="retrieval_query"
                )
                logger.info(f"[OK] Gemini embedding service ready: {self.config.EMBEDDING_MODEL}")
            
            else:
                raise ValueError("No embedding service configured")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize embedding service: {e}")
            raise
    
    def _initialize_answer_generation(self):
        """Initialize answer generation service"""
        try:
            if self.config.OPENAI_API_KEY:
                import openai
                self.answer_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                logger.info(f"[OK] Answer generation ready: {self.config.ANSWER_MODEL}")
            elif self.config.GEMINI_API_KEY:
                import google.generativeai as genai
                self.answer_client = genai
                logger.info("[OK] Gemini answer generation ready")
            else:
                logger.warning("[WARN] No answer generation service configured")
                
        except Exception as e:
            logger.warning(f"[WARN] Could not initialize answer generation: {e}")
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for user query"""
        try:
            if self.embedding_service == "openai":
                response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=query
                )
                return response.data[0].embedding
            
            elif self.embedding_service == "gemini":
                response = self.embedding_client.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
                return response['embedding']
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create query embedding: {e}")
            raise
    
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract keywords from user query for enhanced matching"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'what', 'when', 'where', 'why', 'how',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def perform_semantic_search(self, query: str, 
                               top_k: Optional[int] = None,
                               doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        try:
            logger.info(f"[INFO] Performing semantic search for: '{query}'")
            
            # Create query embedding
            query_embedding = self.create_query_embedding(query)
            
            # Extract keywords for additional filtering
            query_keywords = self.extract_keywords_from_query(query)
            logger.info(f"[INFO] Extracted keywords: {query_keywords}")
            
            # Set search parameters
            if top_k is None:
                top_k = self.config.TOP_K_CHUNKS
            
            # Prepare filters
            search_filter = None
            if doc_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_filter))]
                )
            
            # Perform vector search
            try:
                # Try newer query_points method
                search_results = self.qdrant_client.query_points(
                    collection_name=self.config.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=self.config.SIMILARITY_THRESHOLD,
                    with_payload=True,
                    with_vectors=False
                )
                results = search_results.points
            except AttributeError:
                # Fallback to older search method
                results = self.qdrant_client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=self.config.SIMILARITY_THRESHOLD,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Process and enhance results
            processed_results = []
            for result in results:
                chunk_data = {
                    "id": result.id,
                    "score": result.score,
                    "confidence": self._calculate_confidence(result.score),
                    "chunk_id": result.payload["chunk_id"],
                    "doc_name": result.payload["doc_name"],
                    "doc_id": result.payload["doc_id"],
                    "page": result.payload["page"],
                    "paragraph_number": result.payload["paragraph_number"],
                    "text": result.payload["text"],
                    "keywords": result.payload.get("keywords", []),
                    "char_count": result.payload.get("char_count", 0),
                    "word_count": result.payload.get("word_count", 0),
                    "created_at": result.payload.get("created_at", ""),
                    "embedding_model": result.payload.get("embedding_model", ""),
                }
                
                # Calculate keyword overlap bonus
                chunk_keywords = chunk_data["keywords"]
                if chunk_keywords and query_keywords:
                    keyword_overlap = len(set(query_keywords) & set(chunk_keywords))
                    keyword_bonus = keyword_overlap / len(query_keywords) * 0.1  # 10% bonus max
                    chunk_data["keyword_bonus"] = keyword_bonus
                    chunk_data["adjusted_score"] = chunk_data["score"] + keyword_bonus
                else:
                    chunk_data["keyword_bonus"] = 0.0
                    chunk_data["adjusted_score"] = chunk_data["score"]
                
                processed_results.append(chunk_data)
            
            # Sort by adjusted score (semantic + keyword matching)
            processed_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
            logger.info(f"[OK] Found {len(processed_results)} relevant chunks")
            return processed_results
            
        except Exception as e:
            logger.error(f"[ERROR] Search failed: {e}")
            return []
    
    def select_best_chunks_for_answer(self, search_results: List[Dict[str, Any]], 
                                    query: str) -> List[Dict[str, Any]]:
        """Select the best chunks for answer generation using advanced ranking"""
        
        if not search_results:
            return []
        
        logger.info(f"[INFO] Selecting best chunks from {len(search_results)} candidates")
        
        # Take top percentage of chunks based on similarity score
        top_percent = 0.3  # Top 30% of results
        top_count = max(1, int(len(search_results) * top_percent))
        top_chunks = search_results[:top_count]
        
        # Advanced ranking considering multiple factors
        for chunk in top_chunks:
            rank_score = 0.0
            
            # 1. Base similarity score (70% weight)
            rank_score += chunk["adjusted_score"] * 0.7
            
            # 2. Text length bonus (longer chunks often have more context)
            text_length_norm = min(chunk["char_count"] / 1000, 1.0)  # Normalize to 1000 chars
            rank_score += text_length_norm * 0.1
            
            # 3. Keyword density bonus
            if chunk["keywords"]:
                query_keywords = self.extract_keywords_from_query(query)
                if query_keywords:
                    keyword_density = len(chunk["keywords"]) / len(query_keywords)
                    rank_score += min(keyword_density, 1.0) * 0.1
            
            # 4. Document diversity (prefer chunks from different documents)
            rank_score += 0.05  # Base diversity bonus
            
            # 5. Recency bonus (if available)
            if chunk.get("created_at"):
                rank_score += 0.05
            
            chunk["final_rank_score"] = rank_score
        
        # Sort by final ranking score
        top_chunks.sort(key=lambda x: x["final_rank_score"], reverse=True)
        
        # Select final chunks ensuring diversity
        selected_chunks = []
        used_docs = set()
        
        for chunk in top_chunks:
            # Ensure document diversity
            if len(selected_chunks) < self.config.FINAL_CHUNKS:
                selected_chunks.append(chunk)
                used_docs.add(chunk["doc_name"])
            elif chunk["doc_name"] not in used_docs and len(selected_chunks) < self.config.FINAL_CHUNKS + 2:
                selected_chunks.append(chunk)
                used_docs.add(chunk["doc_name"])
        
        # Limit to final count
        selected_chunks = selected_chunks[:self.config.FINAL_CHUNKS]
        
        logger.info(f"[OK] Selected {len(selected_chunks)} best chunks for answer generation")
        return selected_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive answer using retrieved chunks"""
        
        if not context_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "confidence": "low",
                "sources": []
            }
        
        try:
            logger.info(f"[INFO] Generating answer using {len(context_chunks)} chunks")
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, chunk in enumerate(context_chunks, 1):
                context_part = f"[Source {i}]\n"
                context_part += f"Document: {chunk['doc_name']}\n"
                context_part += f"Page: {chunk['page']}, Paragraph: {chunk['paragraph_number']}\n"
                context_part += f"Content: {chunk['text']}\n"
                context_part += f"Relevance Score: {chunk['score']:.3f}\n"
                
                context_parts.append(context_part)
                
                sources.append({
                    "doc_name": chunk['doc_name'],
                    "page": chunk['page'],
                    "paragraph": chunk['paragraph_number'],
                    "chunk_id": chunk['chunk_id'],
                    "relevance_score": round(chunk['score'], 3),
                    "keywords": chunk.get('keywords', [])
                })
            
            # Combine context (limit length)
            full_context = "\n\n".join(context_parts)
            if len(full_context) > self.config.MAX_CONTEXT_LENGTH:
                # Truncate context to fit limit
                full_context = full_context[:self.config.MAX_CONTEXT_LENGTH] + "..."
            
            # Create comprehensive prompt
            prompt = f"""You are an expert assistant that provides accurate, comprehensive answers based on provided context.

USER QUESTION: {query}

RELEVANT CONTEXT:
{full_context}

INSTRUCTIONS:
1. Provide a comprehensive and accurate answer based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state what information is missing
3. Include specific references to sources when making claims
4. Structure your answer clearly with key points
5. If there are conflicting information in sources, mention it
6. Be concise but thorough

ANSWER:"""

            # Generate answer
            if self.answer_client and hasattr(self.answer_client, 'chat'):
                # OpenAI
                response = self.answer_client.chat.completions.create(
                    model=self.config.ANSWER_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on given context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                answer = response.choices[0].message.content
                
            elif self.answer_client and hasattr(self.answer_client, 'generate_content'):
                # Gemini
                model = self.answer_client.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                answer = response.text
                
            else:
                # Fallback - create answer from context
                answer = self._create_fallback_answer(query, context_chunks)
            
            # Calculate confidence based on chunk scores and coverage
            avg_score = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)
            confidence = self._calculate_answer_confidence(avg_score, len(context_chunks))
            
            logger.info(f"[OK] Answer generated with {confidence} confidence")
            
            return {
                "answer": answer.strip(),
                "confidence": confidence,
                "sources": sources,
                "context_chunks_used": len(context_chunks),
                "avg_relevance_score": round(avg_score, 3),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Answer generation failed: {e}")
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "confidence": "error",
                "sources": sources if 'sources' in locals() else []
            }
    
    def _create_fallback_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a basic answer when AI generation is not available"""
        
        # Combine relevant text from chunks
        relevant_texts = []
        for chunk in chunks:
            text = chunk['text']
            if len(text) > 200:
                text = text[:200] + "..."
            relevant_texts.append(f"From {chunk['doc_name']} (Page {chunk['page']}): {text}")
        
        answer = f"Based on the available documents, here's what I found:\n\n"
        answer += "\n\n".join(relevant_texts)
        answer += f"\n\nThis information is compiled from {len(chunks)} relevant document sections."
        
        return answer
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.7:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _calculate_answer_confidence(self, avg_score: float, num_chunks: int) -> str:
        """Calculate overall answer confidence"""
        base_confidence = avg_score
        
        # Bonus for multiple supporting chunks
        if num_chunks >= 3:
            base_confidence += 0.1
        elif num_chunks >= 2:
            base_confidence += 0.05
        
        if base_confidence >= 0.8:
            return "very_high"
        elif base_confidence >= 0.7:
            return "high"
        elif base_confidence >= 0.6:
            return "medium"
        elif base_confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def search_and_answer(self, query: str, 
                         doc_filter: Optional[str] = None,
                         detailed_output: bool = False) -> Dict[str, Any]:
        """Complete search and answer pipeline"""
        
        start_time = time.time()
        
        try:
            # Step 1: Perform semantic search
            search_results = self.perform_semantic_search(query, doc_filter=doc_filter)
            
            if not search_results:
                return {
                    "query": query,
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "confidence": "none",
                    "sources": [],
                    "search_results_count": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Select best chunks
            best_chunks = self.select_best_chunks_for_answer(search_results, query)
            
            # Step 3: Generate answer
            answer_result = self.generate_answer(query, best_chunks)
            
            # Step 4: Compile final result
            result = {
                "query": query,
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "sources": answer_result["sources"],
                "search_results_count": len(search_results),
                "chunks_used_for_answer": len(best_chunks),
                "processing_time": round(time.time() - start_time, 2)
            }
            
            if detailed_output:
                result["all_search_results"] = search_results
                result["selected_chunks"] = best_chunks
                result["search_metadata"] = {
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "collection_name": self.config.COLLECTION_NAME,
                    "similarity_threshold": self.config.SIMILARITY_THRESHOLD
                }
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Search and answer pipeline failed: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "confidence": "error",
                "sources": [],
                "processing_time": time.time() - start_time
            }

def interactive_search_session():
    """Interactive search session for testing"""
    print("=" * 60)
    print("SEMANTIC SEARCH & ANSWER GENERATION")
    print("=" * 60)
    
    try:
        # Initialize search engine
        print("[INFO] Initializing search engine...")
        search_engine = SemanticSearchEngine(config)
        print("[OK] Search engine ready!")
        
        print(f"\n[INFO] Knowledge Base Statistics:")
        collection_info = search_engine.qdrant_client.get_collection(config.COLLECTION_NAME)
        print(f"   Collection: {config.COLLECTION_NAME}")
        print(f"   Total documents: {collection_info.points_count:,}")
        print(f"   Vector dimension: {collection_info.config.params.vectors.size}")
        
        print(f"\n[INFO] Search Settings:")
        print(f"   Top-K retrieval: {config.TOP_K_CHUNKS}")
        print(f"   Final chunks for answer: {config.FINAL_CHUNKS}")
        print(f"   Similarity threshold: {config.SIMILARITY_THRESHOLD}")
        
        print(f"\n" + "=" * 60)
        print("Ask questions about your documents (type 'quit' to exit)")
        print("Commands:")
        print("  - 'help': Show commands")
        print("  - 'stats': Show search statistics")
        print("  - 'detailed <question>': Get detailed search results")
        print("=" * 60)
        
        search_count = 0
        
        while True:
            try:
                query = input("\n[QUESTION] ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit']:
                    print("[INFO] Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\n[HELP] Available commands:")
                    print("  - Ask any question about your documents")
                    print("  - 'detailed <question>': Get detailed search results")
                    print("  - 'stats': Show search statistics")
                    print("  - 'quit': Exit the program")
                    continue
                
                if query.lower() == 'stats':
                    print(f"\n[STATS] Session Statistics:")
                    print(f"   Searches performed: {search_count}")
                    print(f"   Collection: {config.COLLECTION_NAME}")
                    continue
                
                # Check for detailed search request
                detailed = False
                if query.lower().startswith('detailed '):
                    detailed = True
                    query = query[9:]  # Remove 'detailed ' prefix
                
                search_count += 1
                print(f"\n[SEARCH] Processing question {search_count}...")
                
                # Perform search and answer
                start_time = time.time()
                result = search_engine.search_and_answer(query, detailed_output=detailed)
                
                # Display results
                print(f"\n[ANSWER] ({result['confidence']} confidence)")
                print("=" * 50)
                print(result['answer'])
                print("=" * 50)
                
                print(f"\n[SOURCES] ({len(result['sources'])} sources)")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['doc_name']} (Page {source['page']}, Para {source['paragraph']})")
                    print(f"   Relevance: {source['relevance_score']:.3f}")
                    if source['keywords']:
                        print(f"   Keywords: {', '.join(source['keywords'][:5])}")
                
                print(f"\n[METADATA]")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Search results: {result['search_results_count']}")
                print(f"   Chunks used: {result['chunks_used_for_answer']}")
                
                if detailed and 'all_search_results' in result:
                    print(f"\n[DETAILED] All Search Results:")
                    for i, chunk in enumerate(result['all_search_results'][:10], 1):
                        print(f"{i}. Score: {chunk['score']:.3f} | {chunk['doc_name']} (Page {chunk['page']})")
                        print(f"   Text: {chunk['text'][:100]}...")
                
            except KeyboardInterrupt:
                print("\n[INFO] Search interrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
                continue
    
    except Exception as e:
        print(f"[ERROR] Failed to initialize search engine: {e}")
        print("\n[INFO] Troubleshooting tips:")
        print("   1. Make sure Qdrant is running and accessible")
        print("   2. Check your .env file configuration")
        print("   3. Ensure embeddings have been created and stored")
        print("   4. Verify collection name matches your embeddings")

def batch_search_from_file(questions_file: str, output_file: str):
    """Process multiple questions from a file"""
    print(f"[INFO] Processing questions from: {questions_file}")
    
    try:
        search_engine = SemanticSearchEngine(config)
        
        # Load questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"[INFO] Processing question {i}/{len(questions)}: {question[:50]}...")
            
            result = search_engine.search_and_answer(question, detailed_output=True)
            results.append(result)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Results saved to: {output_file}")
        
        # Print summary
        total_time = sum(r['processing_time'] for r in results)
        avg_confidence = Counter(r['confidence'] for r in results)
        
        print(f"\n[SUMMARY] Batch Processing Results:")
        print(f"   Questions processed: {len(questions)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per question: {total_time/len(questions):.2f}s")
        print(f"   Confidence distribution: {dict(avg_confidence)}")
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")

def test_search_quality():
    """Test search quality with sample questions"""
    print("[INFO] Running search quality tests...")
    
    # Sample test questions - customize these for your domain
    test_questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key findings?",
        "What are the important recommendations?",
        "What challenges are mentioned?",
        "What solutions are proposed?",
    ]
    
    try:
        search_engine = SemanticSearchEngine(config)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[TEST {i}] {question}")
            print("-" * 40)
            
            result = search_engine.search_and_answer(question)
            
            print(f"Confidence: {result['confidence']}")
            print(f"Sources: {result['search_results_count']}")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Time: {result['processing_time']:.2f}s")
        
        print(f"\n[OK] Quality test completed")
        
    except Exception as e:
        print(f"[ERROR] Quality test failed: {e}")

def search_with_filters():
    """Interactive search with document filtering"""
    print("[INFO] Document-filtered search mode")
    
    try:
        search_engine = SemanticSearchEngine(config)
        
        # Get available documents
        collection_info = search_engine.qdrant_client.get_collection(config.COLLECTION_NAME)
        print(f"[INFO] Searching collection with {collection_info.points_count:,} chunks")
        
        # Sample search to get document names
        sample_results = search_engine.perform_semantic_search("document", top_k=50)
        doc_names = list(set(result['doc_name'] for result in sample_results))
        
        print(f"\n[INFO] Available documents ({len(doc_names)}):")
        for i, doc_name in enumerate(doc_names[:10], 1):
            print(f"   {i}. {doc_name}")
        
        if len(doc_names) > 10:
            print(f"   ... and {len(doc_names) - 10} more")
        
        while True:
            query = input("\n[QUESTION] ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            
            if not query:
                continue
            
            # Ask for document filter
            doc_filter = input("[FILTER] Enter document name (or press Enter for all): ").strip()
            if not doc_filter:
                doc_filter = None
            
            result = search_engine.search_and_answer(query, doc_filter=doc_filter)
            
            print(f"\n[ANSWER] ({result['confidence']} confidence)")
            print("=" * 50)
            print(result['answer'])
            print("=" * 50)
            
            print(f"\n[SOURCES] ({len(result['sources'])} sources)")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['doc_name']} (Page {source['page']})")
    
    except Exception as e:
        print(f"[ERROR] Filtered search failed: {e}")

def export_search_results(query: str, output_format: str = "json"):
    """Export search results to file"""
    print(f"[INFO] Exporting search results for: '{query}'")
    
    try:
        search_engine = SemanticSearchEngine(config)
        result = search_engine.search_and_answer(query, detailed_output=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == "json":
            filename = f"search_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif output_format.lower() == "txt":
            filename = f"search_results_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"SEARCH QUERY: {query}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"ANSWER ({result['confidence']} confidence):\n")
                f.write(result['answer'] + "\n\n")
                f.write("SOURCES:\n")
                for i, source in enumerate(result['sources'], 1):
                    f.write(f"{i}. {source['doc_name']} (Page {source['page']})\n")
                    f.write(f"   Relevance: {source['relevance_score']:.3f}\n")
                f.write(f"\nMETADATA:\n")
                f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                f.write(f"Search results: {result['search_results_count']}\n")
        
        print(f"[OK] Results exported to: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")

def get_collection_insights():
    """Get insights about the knowledge base"""
    print("[INFO] Analyzing knowledge base...")
    
    try:
        search_engine = SemanticSearchEngine(config)
        
        # Get collection info
        collection_info = search_engine.qdrant_client.get_collection(config.COLLECTION_NAME)
        
        print(f"\n[INSIGHTS] Knowledge Base Analysis:")
        print(f"   Collection: {config.COLLECTION_NAME}")
        print(f"   Total chunks: {collection_info.points_count:,}")
        print(f"   Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
        print(f"   Status: {collection_info.status}")
        
        # Sample data to analyze
        sample_results = search_engine.perform_semantic_search("content analysis", top_k=100)
        
        if sample_results:
            # Document distribution
            doc_counts = Counter(result['doc_name'] for result in sample_results)
            
            print(f"\n[INSIGHTS] Document Distribution (Top 10):")
            for doc_name, count in doc_counts.most_common(10):
                print(f"   {doc_name}: {count} chunks")
            
            # Keyword analysis
            all_keywords = []
            for result in sample_results:
                all_keywords.extend(result.get('keywords', []))
            
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                print(f"\n[INSIGHTS] Top Keywords:")
                for keyword, count in keyword_counts.most_common(15):
                    print(f"   {keyword}: {count}")
            
            # Content statistics
            avg_length = sum(result['char_count'] for result in sample_results) / len(sample_results)
            avg_words = sum(result['word_count'] for result in sample_results) / len(sample_results)
            
            print(f"\n[INSIGHTS] Content Statistics:")
            print(f"   Average chunk length: {avg_length:.0f} characters")
            print(f"   Average word count: {avg_words:.0f} words")
            print(f"   Unique documents: {len(doc_counts)}")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")

def main():
    """Main entry point with multiple modes"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--interactive" or command == "-i":
            interactive_search_session()
            
        elif command == "--test" or command == "-t":
            test_search_quality()
            
        elif command == "--insights" or command == "--analyze":
            get_collection_insights()
            
        elif command == "--filtered" or command == "-f":
            search_with_filters()
            
        elif command == "--batch" or command == "-b":
            if len(sys.argv) >= 4:
                batch_search_from_file(sys.argv[2], sys.argv[3])
            else:
                print("[ERROR] Usage: python search.py --batch <questions_file> <output_file>")
                
        elif command == "--export" or command == "-e":
            if len(sys.argv) >= 3:
                query = " ".join(sys.argv[2:])
                export_search_results(query)
            else:
                print("[ERROR] Usage: python search.py --export <your question>")
                
        elif command == "--help" or command == "-h":
            print("SEMANTIC SEARCH & ANSWER GENERATION")
            print("=" * 40)
            print("Usage: python search.py [command]")
            print("\nCommands:")
            print("  --interactive, -i     Interactive search session (default)")
            print("  --test, -t           Run search quality tests")
            print("  --insights           Analyze knowledge base")
            print("  --filtered, -f       Search with document filters")
            print("  --batch, -b          Process questions from file")
            print("  --export, -e         Export search results")
            print("  --help, -h           Show this help")
            print("\nExamples:")
            print("  python search.py --interactive")
            print("  python search.py --batch questions.txt results.json")
            print("  python search.py --export \"What is the main topic?\"")
            print("\nConfiguration:")
            print(f"  Collection: {config.COLLECTION_NAME}")
            print(f"  Top-K retrieval: {config.TOP_K_CHUNKS}")
            print(f"  Final chunks: {config.FINAL_CHUNKS}")
            print(f"  Similarity threshold: {config.SIMILARITY_THRESHOLD}")
            
        else:
            # Treat as a direct question
            query = " ".join(sys.argv[1:])
            print(f"[QUESTION] {query}")
            
            try:
                search_engine = SemanticSearchEngine(config)
                result = search_engine.search_and_answer(query)
                
                print(f"\n[ANSWER] ({result['confidence']} confidence)")
                print("=" * 50)
                print(result['answer'])
                print("=" * 50)
                
                print(f"\n[SOURCES] ({len(result['sources'])} sources)")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['doc_name']} (Page {source['page']})")
                    
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
    
    else:
        # Default to interactive mode
        interactive_search_session()

if __name__ == "__main__":
    main()
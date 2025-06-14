"""
Concise Theme Identifier - One Comprehensive Answer Per Theme
Groups all related content into distinct themes and provides ONE precise answer for each
No repetitive responses - just clean, comprehensive answers users actually want
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_env_file():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    env_files = [Path('.env'), PROJECT_ROOT / '.env']
    for env_file in env_files:
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env_file()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ConciseThemeIdentifier:
    def __init__(self):
        """
        Concise theme identifier that produces ONE comprehensive answer per theme
        No repetitive or fragmentary responses
        """
        
        self.project_root = PROJECT_ROOT
        self.keywords_dir = self.project_root / "data" / "keywords_enhanced"
        self.embeddings_dir = self.project_root / "data" / "embeddings"
        self.output_dir = self.project_root / "data" / "themes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_key = os.getenv('GROQ_API_KEY', '')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        self.rate_limit_delay = 3.0
        
        # Load data
        self.chunks_data = self._load_chunks()
        self.embeddings_cache = self._load_embeddings()
        
        print(f"✅ Loaded {len(self.chunks_data)} chunks")
        if self.embeddings_cache is not None:
            print(f"✅ Loaded embeddings: {self.embeddings_cache.shape}")
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks data"""
        keywords_file = self.keywords_dir / "all_chunks_with_keywords.json"
        
        if keywords_file.exists():
            try:
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('chunks', [])
            except Exception as e:
                logger.error(f"Error loading chunks: {e}")
        return []
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings if available"""
        embeddings_files = [
            self.embeddings_dir / "embeddings_cache.npy",
            self.project_root / "embeddings_cache.npy"
        ]
        
        for emb_file in embeddings_files:
            if emb_file.exists():
                try:
                    embeddings = np.load(emb_file)
                    if len(embeddings) == len(self.chunks_data):
                        return embeddings
                except Exception:
                    continue
        return None
    
    def check_setup(self) -> bool:
        """Check setup"""
        if not self.api_key:
            print("❌ Set GROQ_API_KEY environment variable")
            return False
        if not self.chunks_data:
            print("❌ No chunks data found")
            return False
        
        print(f"✅ Ready: {len(self.chunks_data)} chunks, API configured")
        return True
    
    def identify_themes_concise(self, query: str, max_themes: int = 5) -> List[Dict]:
        """
        Identify themes and create ONE comprehensive answer per theme
        No redundancy, no repetition - just clean, distinct themes with complete answers
        """
        
        print(f"\n🎯 ANALYZING: '{query}'")
        
        # Step 1: Get all relevant content
        relevant_chunks = self._get_relevant_chunks(query)
        if not relevant_chunks:
            print("❌ No relevant content found")
            return []
        
        print(f"📊 Found {len(relevant_chunks)} relevant chunks")
        
        # Step 2: Group into distinct themes (using best available method)
        themes = self._group_into_themes(relevant_chunks, max_themes)
        if not themes:
            print("❌ No themes could be formed")
            return []
        
        print(f"🧩 Identified {len(themes)} distinct themes")
        
        # Step 3: Create ONE comprehensive answer per theme
        final_themes = []
        for i, theme_chunks in enumerate(themes, 1):
            print(f"   Generating answer for theme {i}...")
            
            comprehensive_theme = self._create_comprehensive_theme(query, theme_chunks, i)
            if comprehensive_theme:
                final_themes.append(comprehensive_theme)
            
            time.sleep(self.rate_limit_delay)
        
        print(f"✅ Generated {len(final_themes)} comprehensive theme answers")
        return final_themes
    
    def _get_relevant_chunks(self, query: str) -> List[Dict]:
        """Get chunks relevant to the query"""
        
        query_words = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_keywords = query_words - stop_words
        
        relevant_chunks = []
        
        for chunk in self.chunks_data:
            # Calculate relevance
            chunk_keywords = set(chunk.get('keywords', []))
            text_words = set(chunk.get('text', '').lower().split())
            
            keyword_match = len(query_keywords & chunk_keywords) / len(query_keywords) if query_keywords else 0
            text_match = len(query_keywords & text_words) / len(query_keywords) if query_keywords else 0
            
            relevance = max(keyword_match, text_match)
            
            if relevance >= 0.1:  # Minimum relevance threshold
                chunk['relevance_score'] = relevance
                relevant_chunks.append(chunk)
        
        # Sort by relevance and limit
        relevant_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return relevant_chunks[:40]  # Limit for processing efficiency
    
    def _group_into_themes(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Group chunks into distinct themes using best available method"""
        
        # Try embeddings-based clustering if available
        if self.embeddings_cache is not None:
            try:
                return self._cluster_with_embeddings(chunks, max_themes)
            except Exception as e:
                logger.warning(f"Embeddings clustering failed: {e}")
        
        # Fallback to keyword-based grouping
        return self._group_by_keywords(chunks, max_themes)
    
    def _cluster_with_embeddings(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Cluster using embeddings"""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise Exception("scikit-learn not available")
        
        # Get embeddings for relevant chunks
        embeddings_subset = []
        valid_chunks = []
        
        for chunk in chunks:
            # Find original chunk index
            for i, orig_chunk in enumerate(self.chunks_data):
                if (orig_chunk.get('doc_id') == chunk.get('doc_id') and
                    orig_chunk.get('page') == chunk.get('page')):
                    embeddings_subset.append(self.embeddings_cache[i])
                    valid_chunks.append(chunk)
                    break
        
        if len(embeddings_subset) < 4:
            raise Exception("Not enough embeddings found")
        
        # Perform clustering
        n_clusters = min(max_themes, len(embeddings_subset) // 3)
        if n_clusters < 2:
            return [valid_chunks]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_subset)
        
        # Group by clusters
        clusters = defaultdict(list)
        for chunk, label in zip(valid_chunks, cluster_labels):
            clusters[label].append(chunk)
        
        # Return clusters with minimum size
        return [cluster for cluster in clusters.values() if len(cluster) >= 2]
    
    def _group_by_keywords(self, chunks: List[Dict], max_themes: int) -> List[List[Dict]]:
        """Group chunks by keyword similarity"""
        
        # Get most common keywords
        all_keywords = []
        for chunk in chunks:
            all_keywords.extend(chunk.get('keywords', []))
        
        top_keywords = [kw for kw, count in Counter(all_keywords).most_common(max_themes * 2)]
        
        # Group by keywords
        groups = []
        used_chunks = set()
        
        for keyword in top_keywords:
            if len(groups) >= max_themes:
                break
                
            group = []
            for i, chunk in enumerate(chunks):
                if i not in used_chunks and keyword in chunk.get('keywords', []):
                    group.append(chunk)
                    used_chunks.add(i)
            
            if len(group) >= 2:  # Minimum group size
                groups.append(group)
        
        # Add remaining high-relevance chunks as a group
        remaining = [chunk for i, chunk in enumerate(chunks) 
                    if i not in used_chunks and chunk.get('relevance_score', 0) >= 0.3]
        
        if remaining and len(remaining) >= 2:
            groups.append(remaining)
        
        return groups
    
    def _create_comprehensive_theme(self, query: str, theme_chunks: List[Dict], theme_id: int) -> Optional[Dict]:
        """
        Create ONE comprehensive theme with a complete answer
        This is the key function that produces clean, non-repetitive responses
        """
        
        if not theme_chunks:
            return None
        
        # Prepare all content from this theme
        all_texts = []
        citations = []
        common_keywords = []
        
        for chunk in theme_chunks:
            all_texts.append(chunk.get('text', ''))
            common_keywords.extend(chunk.get('keywords', []))
            
            citations.append({
                "doc_id": chunk.get('doc_id'),
                "page": chunk.get('page'),
                "paragraph_number": chunk.get('paragraph_number', chunk.get('para_id')),
                "relevance_score": chunk.get('relevance_score', 0)
            })
        
        # Get top keywords for this theme
        top_keywords = [kw for kw, count in Counter(common_keywords).most_common(5)]
        
        # Combine all text content (limit for prompt efficiency)
        combined_content = "\n\n".join(all_texts[:8])  # Top 8 chunks
        
        # Create comprehensive analysis prompt
        prompt = f"""You are analyzing documents to answer the query: "{query}"

THEME CONTENT (related passages about one specific aspect):
{combined_content}

TASK: Create ONE comprehensive, definitive answer about this specific theme.

Requirements:
1. THEME_TITLE: Clear, specific title (4-6 words) that captures this distinct aspect
2. COMPREHENSIVE_ANSWER: One complete, detailed response that:
   - Directly addresses this aspect of the user's query
   - Synthesizes ALL the information from the passages above
   - Provides specific details, examples, and insights
   - Is complete enough that the user needs no additional explanation for this theme
   - Flows naturally like an expert explanation
   - Is 150-250 words

3. Keep it focused on ONE distinct theme - don't try to cover everything

JSON format:
{{
    "theme_title": "Specific Theme Title",
    "comprehensive_answer": "One complete, detailed explanation that fully covers this theme with specific details from the documents. This should be a thorough, standalone answer that synthesizes all the relevant information about this particular aspect of the query."
}}"""
        
        # Get LLM response
        response = self._call_llm_safe(prompt)
        
        if response:
            theme_data = self._parse_theme_response(response, citations, theme_chunks, theme_id)
            if theme_data:
                # Add metadata
                theme_data['citations'] = citations
                theme_data['cluster_size'] = len(theme_chunks)
                theme_data['top_keywords'] = top_keywords
                theme_data['method'] = 'comprehensive_synthesis'
                return theme_data
        
        # Fallback if LLM fails
        return self._create_fallback_theme(theme_chunks, citations, theme_id)
    
    def _call_llm_safe(self, prompt: str) -> Optional[str]:
        """Safe LLM calling with basic error handling"""
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:
                print("   ⏳ Rate limit - waiting...")
                time.sleep(10)
                return self._call_llm_safe(prompt)
            else:
                logger.warning(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None
    
    def _parse_theme_response(self, response: str, citations: List[Dict], 
                             chunks: List[Dict], theme_id: int) -> Optional[Dict]:
        """Parse LLM response into theme structure"""
        
        try:
            # Try JSON parsing
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                
                if 'theme_title' in data and 'comprehensive_answer' in data:
                    return {
                        'theme_title': data['theme_title'],
                        'comprehensive_answer': data['comprehensive_answer'],
                        'confidence': 'high',
                        'answer_length': len(data['comprehensive_answer'].split())
                    }
            
            # Manual parsing fallback
            lines = response.split('\n')
            theme_title = f"Theme {theme_id}"
            comprehensive_answer = "Analysis of related document content."
            
            for line in lines:
                if 'title' in line.lower() and ':' in line:
                    theme_title = line.split(':', 1)[1].strip().strip('"')
                elif 'answer' in line.lower() and ':' in line:
                    comprehensive_answer = line.split(':', 1)[1].strip().strip('"')
                elif len(line.strip()) > 100:  # Likely the main answer
                    comprehensive_answer = line.strip()
            
            return {
                'theme_title': theme_title,
                'comprehensive_answer': comprehensive_answer,
                'confidence': 'medium',
                'answer_length': len(comprehensive_answer.split())
            }
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return None
    
    def _create_fallback_theme(self, chunks: List[Dict], citations: List[Dict], theme_id: int) -> Dict:
        """Create fallback theme when LLM is unavailable"""
        
        # Extract key information
        documents = set(chunk.get('doc_id') for chunk in chunks)
        keywords = []
        for chunk in chunks:
            keywords.extend(chunk.get('keywords', []))
        
        top_keywords = [kw for kw, count in Counter(keywords).most_common(3)]
        
        # Create basic comprehensive answer
        sample_text = ""
        for chunk in chunks[:3]:
            text = chunk.get('text', '')
            if text:
                sample_text += text[:150] + " "
        
        comprehensive_answer = f"This theme covers {', '.join(top_keywords)} based on analysis of {len(chunks)} document segments. "
        comprehensive_answer += f"The content discusses: {sample_text.strip()[:300]}..."
        comprehensive_answer += f" Information is sourced from {len(documents)} documents."
        
        return {
            'theme_title': f"Theme: {', '.join(top_keywords[:2])}",
            'comprehensive_answer': comprehensive_answer,
            'confidence': 'medium',
            'answer_length': len(comprehensive_answer.split())
        }
    
    def display_concise_results(self, themes: List[Dict], query: str):
        """Display clean, concise results - one answer per theme"""
        
        if not themes:
            print("❌ No themes identified")
            return
        
        print(f"\n🎯 ANALYSIS RESULTS FOR: '{query}'")
        print(f"📊 {len(themes)} distinct themes identified")
        print("=" * 80)
        
        for i, theme in enumerate(themes, 1):
            theme_title = theme.get('theme_title', f'Theme {i}')
            comprehensive_answer = theme.get('comprehensive_answer', 'No answer available')
            answer_length = theme.get('answer_length', 0)
            citations_count = len(theme.get('citations', []))
            
            print(f"\n🧩 THEME {i}: {theme_title}")
            print(f"\n💬 ANSWER:")
            print(f"{comprehensive_answer}")
            
            print(f"\n📊 Details: {answer_length} words | {citations_count} sources")
            
            # Show top sources
            citations = theme.get('citations', [])
            if citations:
                top_sources = citations[:3]
                sources_text = ", ".join([f"{c.get('doc_id', 'Unknown')}" for c in top_sources])
                print(f"📄 Key Sources: {sources_text}")
                if len(citations) > 3:
                    print(f"   + {len(citations) - 3} more sources")
            
            print("-" * 60)
        
        # Summary
        total_words = sum(theme.get('answer_length', 0) for theme in themes)
        total_sources = sum(len(theme.get('citations', [])) for theme in themes)
        
        print(f"\n📋 SUMMARY:")
        print(f"   🎯 {len(themes)} comprehensive answers")
        print(f"   📝 {total_words:,} total words")
        print(f"   📚 {total_sources} total sources")
        print(f"   ✅ Complete coverage of query aspects")
    
    def process_query_concise(self, query: str, max_themes: int = 5, save_results: bool = True) -> List[Dict]:
        """Process query and return concise, comprehensive themes"""
        
        start_time = time.time()
        
        # Identify themes
        themes = self.identify_themes_concise(query, max_themes)
        
        # Save results if requested
        if save_results and themes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
            
            result_data = {
                "query": query,
                "processed_at": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - start_time,
                "method": "concise_comprehensive_themes",
                "total_themes": len(themes),
                "themes": themes,
                "summary": {
                    "total_words": sum(theme.get('answer_length', 0) for theme in themes),
                    "total_sources": sum(len(theme.get('citations', [])) for theme in themes),
                    "avg_answer_length": sum(theme.get('answer_length', 0) for theme in themes) / len(themes) if themes else 0
                }
            }
            
            output_file = self.output_dir / f"concise_analysis_{query_safe}_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved to: {output_file}")
        
        return themes

def main():
    """Main function for concise theme identification"""
    
    print("🎯 CONCISE THEME IDENTIFIER")
    print("📝 One Comprehensive Answer Per Theme")
    print("🚫 No Repetition, No Fragments")
    print("=" * 50)
    
    # Initialize
    identifier = ConciseThemeIdentifier()
    
    if not identifier.check_setup():
        return
    
    print(f"\n✨ WHAT YOU GET:")
    print(f"   🎯 Clear, distinct themes")
    print(f"   💬 One comprehensive answer per theme")
    print(f"   📚 Complete source citations")
    print(f"   🚫 No repetitive content")
    
    # Interactive mode
    while True:
        query = input(f"\nEnter your query (or 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            print("❌ Please enter a valid query")
            continue
        
        try:
            start_time = time.time()
            
            # Process query
            themes = identifier.process_query_concise(query)
            
            end_time = time.time()
            
            # Display results
            identifier.display_concise_results(themes, query)
            
            print(f"\n⏱️  Completed in {end_time - start_time:.1f} seconds")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
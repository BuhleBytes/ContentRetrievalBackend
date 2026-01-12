"""
Flask API for Intelligent Content Retrieval System

Provides semantic and hybrid search capabilities with optional AI-powered
text enhancement using Claude API. Implements intelligent caching to
minimize costs while maintaining performance.

Author: Buhle Mlandu
"""

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import llm_enhancer
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List
import os
import hashlib
import json
import time

app = Flask(__name__)
load_dotenv()
CORS(app)

print("✓ Flask app created!")

# Global variables for model and database
model = None
collection = None

# Cache configuration
CACHE_TTL = 3600
CACHE_MAX_SIZE = 100
result_cache: Dict[str, tuple] = {}

print("✓ Cache configured (TTL: 1 hour, Max: 100 queries)")


def generate_cache_key(
    query_text: str, 
    n_results: int, 
    filter_category: str = None,
    enable_llm: bool = False
) -> str:
    """
    Generate unique cache key from request parameters.
    
    The cache key includes the LLM flag to ensure enhanced and non-enhanced
    results are cached separately.
    """
    params = {
        'query': query_text,
        'n_results': n_results,
        'filter_category': filter_category,
        'enable_llm': enable_llm
    }
    
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    
    return cache_key


def generate_hybrid_cache_key(
    query_text: str, 
    keywords: List[str], 
    n_results: int, 
    filter_category: str = None,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    enable_llm: bool = False
) -> str:
    """
    Generate unique cache key for hybrid search.
    
    Includes all parameters that affect search results to ensure
    proper cache differentiation.
    """
    params = {
        'query': query_text,
        'keywords': sorted(keywords) if keywords else [],
        'n_results': n_results,
        'filter_category': filter_category,
        'semantic_weight': semantic_weight,
        'keyword_weight': keyword_weight,
        'enable_llm': enable_llm
    }
    
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    
    return cache_key


def get_cached_result(cache_key: str) -> Dict | None:
    """
    Retrieve result from cache if it exists and hasn't expired.
    """
    if cache_key not in result_cache:
        return None
    
    result, timestamp = result_cache[cache_key]
    current_time = time.time()
    
    if current_time - timestamp > CACHE_TTL:
        del result_cache[cache_key]
        return None
    
    return result


def set_cached_result(cache_key: str, result: Dict) -> None:
    """
    Store result in cache with current timestamp.
    Implements FIFO eviction when cache reaches maximum size.
    """
    global result_cache
    
    if len(result_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(result_cache))
        del result_cache[oldest_key]
    
    result_cache[cache_key] = (result, time.time())


def clear_cache() -> None:
    """Clear all cached results"""
    global result_cache
    result_cache = {}


def get_cache_stats() -> Dict:
    """Calculate and return current cache statistics"""
    current_time = time.time()
    valid_entries = 0
    expired_entries = 0
    
    for cache_key, (result, timestamp) in result_cache.items():
        if current_time - timestamp > CACHE_TTL:
            expired_entries += 1
        else:
            valid_entries += 1
    
    return {
        'total_entries': len(result_cache),
        'valid_entries': valid_entries,
        'expired_entries': expired_entries,
        'max_size': CACHE_MAX_SIZE,
        'ttl_seconds': CACHE_TTL
    }


print("✓ Cache functions defined!")


def initialize_system():
    """
    Initialize the embedding model and ChromaDB database.
    Also attempts to initialize the Claude API for LLM enhancement.
    """
    global model, collection
   
    print("\n🚀 INITIALIZING INTELLIGENT CONTENT RETRIEVAL SYSTEM\n")
    
    try:
        print("Step 1: Loading embedding model")
        model_name = 'all-mpnet-base-v2'
        model = SentenceTransformer(model_name)

        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"   ✓ Model loaded: {model_name}")
        print(f"   ✓ Embedding dimensions: {embedding_dim}D")

        print("\n🗄️  Step 2: Loading ChromaDB...")
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chromadb')

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found at '{db_path}'!\n"
                f"Please run Part 4 first to create the database."
            )
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="intelligent_content_retrieval")

        doc_count = collection.count()
        print(f"   ✓ Database loaded: {db_path}")
        print(f"   ✓ Collection: intelligent_content_retrieval")
        print(f"   ✓ Documents: {doc_count}")

        print("\n✅ SYSTEM READY!\n")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution:")
        print("   1. Make sure you've run Part 4 (part04VectorDB.py)")
        print("   2. Check that data/chromadb/ exists")
        print("   3. Run the API from your project root directory")
        raise

    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {type(e).__name__}: {e}")
        raise

    try:
        llm_enhancer.initialize_claude()
    except ValueError as e:
        print(f"\n⚠️  Claude API not available: {e}")
        print("   LLM enhancement feature will be disabled")


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculate the proportion of keywords found in the text.
    Returns a score between 0.0 and 1.0.
    """
    if not keywords or len(keywords) == 0:
        return 0.0
    
    text_lower = text.lower()
    matches = 0
    
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in text_lower:
            matches += 1
    
    return matches / len(keywords)


@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        'name': 'Intelligent Content Retrieval System API',
        'version': '1.0.0',
        'author': 'Buhle Mlandu',
        'description': 'Semantic search API powered by ChromaDB and Sentence Transformers',
        'endpoints': {
            '/': 'API information (you are here)',
            '/health': 'Check system health',
            '/stats': 'Get database statistics',
            '/search': 'Semantic search (POST)',
            '/hybrid': 'Hybrid search (POST)',
            '/enhance': 'Enhance specific chunks (POST)'
        },
        'status': 'operational'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    System health check endpoint.
    Returns status of model, database, and document count.
    """
    try:
        model_status = model is not None
        db_status = collection is not None
        doc_count = collection.count() if collection else 0
        
        is_healthy = model_status and db_status
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': model_status,
            'database_loaded': db_status,
            'document_count': doc_count,
            'model_name': 'all-mpnet-base-v2',
            'embedding_dimensions': 768
        })
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_statistics():
    """
    Database statistics endpoint.
    Returns document counts by category and database configuration.
    """
    try:
        if not collection:
            return jsonify({'error': 'Database not initialized'}), 500
        
        sample = collection.peek(collection.count())
        
        categories = {}
        for metadata in sample['metadatas']:
            cat = metadata.get('source_category', 'Unknown')
            if cat not in categories:
                categories[cat] = {
                    'count': 0,
                    'url': metadata.get('source_url', 'N/A')
                }
            categories[cat]['count'] += 1
        
        return jsonify({
            'total_documents': collection.count(),
            'model': 'all-mpnet-base-v2',
            'dimensions': 768,
            'distance_metric': 'cosine',
            'categories': categories
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def semantic_search():
    """
    Semantic search endpoint with optional AI enhancement.
    
    Request body:
        query (str): Search query
        n_results (int): Number of results (1-20, default: 5)
        filter_category (str): Optional category filter
        enable_llm (bool): Enable AI text enhancement (default: false)
    
    Returns enhanced results for first 5 chunks when LLM is enabled.
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query',
                'example': {
                    'query': 'What is machine learning?',
                    'n_results': 5,
                    'enable_llm': True
                }
            }), 400
        
        query_text = data['query']
        n_results = data.get('n_results', 5)
        filter_category = data.get('filter_category', None)
        enable_llm = data.get('enable_llm', False)
        
        if not isinstance(n_results, int) or n_results < 1 or n_results > 20:
            return jsonify({'error': 'n_results must be between 1 and 20'}), 400
        
        cache_key = generate_cache_key(
            query_text, 
            n_results, 
            filter_category,
            enable_llm
        )
        
        cached_result = get_cached_result(cache_key)

        if cached_result is not None:
            print(f"\n🎯 CACHE HIT: '{query_text}' (LLM: {enable_llm})")
            cached_result['cached'] = True
            return jsonify(cached_result)
        
        print(f"\n🔍 SEARCH REQUEST:")
        print(f"   Query: '{query_text}'")
        print(f"   Results: {n_results}")
        print(f"   LLM Enhancement: {enable_llm}")
        if filter_category:
            print(f"   Filter: {filter_category}")
        
        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"   ✓ Generated embedding: {query_embedding.shape}")
        
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results
        }
        
        if filter_category:
            query_params["where"] = {"source_category": filter_category}
        
        results = collection.query(**query_params)
        
        print(f"   ✓ Found {len(results['documents'][0])} results")
        
        formatted_results = []
        
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            
            formatted_results.append({
                'text': doc,
                'similarity': round(similarity, 4),
                'metadata': {
                    'category': metadata.get('source_category', 'N/A'),
                    'url': metadata.get('source_url', 'N/A'),
                    'domain': metadata.get('source_domain', 'N/A'),
                    'chunk_index': metadata.get('chunk_index', 'N/A'),
                    'total_chunks': metadata.get('total_chunks_from_source', 'N/A')
                }
            })
        
        enhancement_info = None
        if enable_llm:
            try:
                enhancement_result = llm_enhancer.smart_enhance_results(
                    formatted_results,
                    query_text,
                    auto_enhance_limit=5
                )
                
                formatted_results = enhancement_result['results']
                enhancement_info = {
                    'auto_enhanced': enhancement_result['auto_enhanced'],
                    'pending_enhancement': enhancement_result['pending_enhancement'],
                    'total_tokens': enhancement_result['total_tokens'],
                    'cost_estimate': enhancement_result['cost_estimate'],
                    'relevance_summary': enhancement_result['relevance_summary']
                }
                
            except Exception as e:
                print(f"   ⚠️ Enhancement failed: {e}")
                enhancement_info = {
                    'error': str(e),
                    'auto_enhanced': 0,
                    'pending_enhancement': len(formatted_results)
                }
        
        response = {
            'query': query_text,
            'search_mode': 'semantic',
            'results': formatted_results,
            'count': len(formatted_results),
            'cached': False,
            'llm_enabled': enable_llm,
            'enhancement_info': enhancement_info
        }

        set_cached_result(cache_key, response)
        print(f"   ✓ Cached result")

        return jsonify(response)
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/enhance', methods=['POST'])
def enhance_remaining():
    """
    On-demand enhancement endpoint for specific chunks.
    
    Used when user wants to enhance results beyond the first 5.
    
    Request body:
        query (str): Original search query
        results (list): Full results array
        indices (list): Indices of chunks to enhance
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'results' not in data or 'indices' not in data:
            return jsonify({
                'error': 'Missing required fields: query, results, indices'
            }), 400
        
        query_text = data['query']
        results = data['results']
        indices = data['indices']
        
        print(f"\n🎯 On-Demand Enhancement Request:")
        print(f"   Query: '{query_text}'")
        print(f"   Enhancing indices: {indices}")
        
        enhancement_result = llm_enhancer.enhance_specific_chunks(
            results,
            indices,
            query_text
        )
        
        return jsonify(enhancement_result)
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/hybrid', methods=['POST'])
def hybrid_search():
    """
    Hybrid search endpoint combining semantic similarity with keyword matching.
    
    Request body:
        query (str): Search query
        keywords (list): List of keywords for filtering
        n_results (int): Number of results (1-20, default: 5)
        semantic_weight (float): Weight for semantic score (0-1, default: 0.7)
        keyword_weight (float): Weight for keyword score (0-1, default: 0.3)
        filter_category (str): Optional category filter
        enable_llm (bool): Enable AI text enhancement (default: false)
    
    Weights must sum to 1.0.
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error':'Missing required field: query'}), 400
        
        query_text = data['query']
        keywords = data.get('keywords', [])
        n_results = data.get('n_results', 5)
        filter_category = data.get('filter_category', None)
        semantic_weight = data.get('semantic_weight', 0.7)
        keyword_weight = data.get('keyword_weight', 0.3)
        enable_llm = data.get('enable_llm', False)

        if not (0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1):
            return jsonify({'error': 'Weights must be between 0 and 1'}), 400
        
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        cache_key = generate_hybrid_cache_key(
            query_text,
            keywords,
            n_results,
            filter_category,
            semantic_weight,
            keyword_weight,
            enable_llm
        )
        
        cached_result = get_cached_result(cache_key)

        if cached_result is not None:
            print(f"\n🎯 CACHE HIT (HYBRID): '{query_text}' (LLM: {enable_llm})")
            cached_result['cached'] = True
            return jsonify(cached_result)
        
        print(f"\n❌ CACHE MISS (HYBRID): '{query_text}'")
        print(f"\n🔍 HYBRID SEARCH REQUEST:")
        print(f"   Query: '{query_text}'")
        print(f"   Keywords: {keywords}")
        print(f"   Results requested: {n_results}")
        print(f"   Weights: {semantic_weight*100:.0f}% semantic + {keyword_weight*100:.0f}% keyword")
        print(f"   LLM Enhancement: {enable_llm}")

        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        retrieval_count = min(n_results * 10, 100)

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": retrieval_count
        }
        
        if filter_category:
            query_params["where"] = {"source_category": filter_category}

        semantic_results = collection.query(**query_params)
        print(f"   ✓ Retrieved {retrieval_count} candidates for re-ranking")

        hybrid_results = []
        
        for doc, metadata, distance in zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0],
            semantic_results['distances'][0]
        ):
            semantic_score = 1 - distance
            keyword_score = calculate_keyword_score(doc, keywords) if keywords else 0.0
            hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            hybrid_results.append({
                'text': doc,
                'hybrid_score': round(hybrid_score, 4),
                'semantic_score': round(semantic_score, 4),
                'keyword_score': round(keyword_score, 4),
                'metadata': {
                    'category': metadata.get('source_category', 'N/A'),
                    'url': metadata.get('source_url', 'N/A'),
                    'domain': metadata.get('source_domain', 'N/A'),
                    'chunk_index': metadata.get('chunk_index', 'N/A'),
                    'total_chunks': metadata.get('total_chunks_from_source', 'N/A')
                }
            })
        
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = hybrid_results[:n_results]
        
        print(f"   ✓ Ranked {len(hybrid_results)} candidates")
        print(f"   ✓ Returning top {len(top_results)} results")
        
        enhancement_info = None
        if enable_llm:
            try:
                enhancement_result = llm_enhancer.smart_enhance_results(
                    top_results,
                    query_text,
                    auto_enhance_limit=5
                )
                
                top_results = enhancement_result['results']
                enhancement_info = {
                    'auto_enhanced': enhancement_result['auto_enhanced'],
                    'pending_enhancement': enhancement_result['pending_enhancement'],
                    'total_tokens': enhancement_result['total_tokens'],
                    'cost_estimate': enhancement_result['cost_estimate'],
                }
                
            except Exception as e:
                print(f"   ⚠️ Enhancement failed: {e}")
                enhancement_info = {
                    'error': str(e),
                    'auto_enhanced': 0,
                    'pending_enhancement': len(top_results)
                }
        
        response = {
            'query': query_text,
            'keywords': keywords,
            'search_mode': 'hybrid',
            'weights': {
                'semantic': semantic_weight,
                'keyword': keyword_weight
            },
            'results': top_results,
            'count': len(top_results),
            'candidates_evaluated': retrieval_count,
            'cached': False,
            'llm_enabled': enable_llm,
            'enhancement_info': enhancement_info
        }
        
        set_cached_result(cache_key, response)
        print(f"   ✓ Cached result")
        
        return jsonify(response)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/cache/stats', methods=['GET'])
def cache_statistics():
    """Get current cache statistics"""
    try:
        stats = get_cache_stats()
        return jsonify({
            'cache_stats': stats,
            'status': 'operational'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    """Clear all cached results"""
    try:
        entries_before = len(result_cache)
        clear_cache()
        return jsonify({
            'message': 'Cache cleared successfully',
            'entries_cleared': entries_before,
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation',
        'available_endpoints': ['/', '/health', '/stats', '/search', '/hybrid', '/enhance']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


initialize_system()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING INTELLIGENT CONTENT RETRIEVAL SYSTEM API")
    print("="*70 + "\n")
    
    print("\n🌐 Starting Flask API server...")
    print("="*70)
    print("📍 API URL: http://localhost:5000")
    print("📚 Endpoints:")
    print("   • GET  /         - API information")
    print("   • GET  /health   - Health check")
    print("   • GET  /stats    - Database statistics")
    print("   • POST /search   - Semantic search")
    print("   • POST /hybrid   - Hybrid search")
    print("   • POST /enhance  - Enhance specific chunks")
    print("="*70)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    port = int(os.getenv('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
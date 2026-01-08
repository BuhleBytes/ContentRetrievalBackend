"""
Flask API for Intelligent Content Retrieval System
Building this step by step !

Author: Buhle Mlandu
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List
import os
import hashlib
import json
import time

#Create Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows web browsers to access the API
CORS(app)
print("✓ Flask app created!")


model = None
collection = None

CACHE_TTL = 3600 # 1 hour (in seconds )
CACHE_MAX_SIZE = 100  # Store max 100 unique queries

#Cache storage: {cache_key: (result, timestamp)}
result_cache: Dict[str, tuple] = {}
print("✓ Cache configured (TTL: 1 hour, Max: 100 queries)")

def generate_cache_key(query_text: str, n_results: int, filter_category: str = None) -> str:
    """
    Generate unique cache key from request parameters.
    
    Args:
        query_text: The search query
        n_results: Number of results requested
        filter_category: Optional category filter
        
    Returns:
        MD5 hash of parameters (32 characters)
    """
    params = {
        'query': query_text,
        'n_results': n_results,
        'filter_category': filter_category
    }
    
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    
    return cache_key

def get_cached_result(cache_key: str) -> Dict | None:
    """
    Get result from cache if it exists and is not expired.
    
    Args:
        cache_key: The cache key to lookup
        
    Returns:
        Cached result dict if found and valid, None otherwise
    """
    if cache_key not in result_cache:
        return None
    
    result, timestamp = result_cache[cache_key]
    current_time = time.time()
    
    # Check if expired
    if current_time - timestamp > CACHE_TTL:
        del result_cache[cache_key]
        return None
    
    return result

def set_cached_result(cache_key: str, result: Dict) -> None:
    """
    Store result in cache with current timestamp.
    
    Implements FIFO eviction when cache is full.
    
    Args:
        cache_key: The cache key
        result: The result to cache
    """
    global result_cache
    
    # If cache is full, remove oldest entry
    if len(result_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(result_cache))
        del result_cache[oldest_key]
    
    result_cache[cache_key] = (result, time.time())

def clear_cache() -> None:
    """Clear entire cache"""
    global result_cache
    result_cache = {}

def get_cache_stats() -> Dict:
    """Get cache statistics"""

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

def generate_hybrid_cache_key(
    query_text: str, 
    keywords: List[str], 
    n_results: int, 
    filter_category: str = None,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> str:
    """
    Generate unique cache key for hybrid search.
    
    IMPORTANT: Must include ALL parameters that affect results!
    """
    params = {
        'query': query_text,
        'keywords': sorted(keywords) if keywords else [],  # Sort for consistency!
        'n_results': n_results,
        'filter_category': filter_category,
        'semantic_weight': semantic_weight,
        'keyword_weight': keyword_weight
    }
    
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    
    return cache_key



def initialize_system():
    """Load the embedding model and ChromaDB database"""
    global model, collection
   
    print("\n🚀 INITIALIZING INTELLIGENT CONTENT RETRIEVAL SYSTEM\n")
    try:
        print("\n Step 1: Loading embedding model")
        model_name = 'all-mpnet-base-v2' #Same model
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
        # Connect to YOUR existing database
        client = chromadb.PersistentClient(path=db_path)

        #Get YOUR collection created in Part 4
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

def calculate_keyword_score(text: str, keywords: List[str]) -> float:
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
    """
    Home page - provides information about the API
    """
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
            '/hybrid': 'Hybrid search (POST)'
        },
        'status': 'operational'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Check if the API is working properly.
    Returns system status and statistics.
    
    Example response:
    {
        "status": "healthy",
        "model_loaded": true,
        "database_loaded": true,
        "document_count": 251
    }
    """
    try:
        # Check if model and database are loaded
        model_status = model is not None
        db_status = collection is not None
        doc_count = collection.count() if collection else 0
        
        # Determine overall health
        is_healthy = model_status and db_status
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': model_status,
            'database_loaded': db_status,
            'document_count': doc_count,
            'model_name': 'all-mpnet-base-v2',
            'updated':'updated'
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
    Get detailed database statistics.
    Shows how many documents are in each category.
    
    Example response:
    {
        "total_documents": 251,
        "model": "all-mpnet-base-v2",
        "categories": {
            "News": {"count": 45, "url": "https://..."},
            "Educational": {"count": 89, "url": "https://..."}
        }
    }
    """
    try:
        if not collection:
            return jsonify({'error': 'Database not initialized'}), 500
        
        # Get sample data to analyze categories
        sample = collection.peek(limit=50)
        
        # Count documents by category
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
    Perform semantic search - this is YOUR Part 5 code as an HTTP endpoint!
    
    REQUEST (send as JSON):
    {
        "query": "What is machine learning?",
        "n_results": 5,
        "filter_category": "Educational"  // optional
    }
    
    RESPONSE:
    {
        "query": "What is machine learning?",
        "results": [
            {
                "text": "Machine learning is...",
                "similarity": 0.8734,
                "metadata": {...}
            }
        ],
        "count": 5
    }
    """
    try:
        # ----------------------------------------------------------------
        # STEP 1: Get data from HTTP request
        # ----------------------------------------------------------------
        data = request.get_json()
        
        # Validate input
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query',
                'example': {
                    'query': 'What is machine learning?',
                    'n_results': 5
                }
            }), 400
        
        query_text = data['query']
        n_results = data.get('n_results', 5)  # Default: 5 results
        filter_category = data.get('filter_category', None)  # Optional filter
        
        # Validate n_results
        if not isinstance(n_results, int) or n_results < 1 or n_results > 20:
            return jsonify({'error': 'n_results must be between 1 and 20'}), 400
        
        # CACHE CHECK
        cache_key = generate_cache_key(query_text, n_results, filter_category)
        cached_result = get_cached_result(cache_key)

        if cached_result is not None:
            print(f"\n🎯 CACHE HIT: '{query_text}' (key: {cache_key[:8]}...)")
            cached_result['cached'] = True
            return jsonify(cached_result)
        
        print(f"\n❌ CACHE MISS: '{query_text}' (key: {cache_key[:8]}...)")
        print(f"   Query: '{query_text}'")
        print(f"   Results: {n_results}")

        print(f"\n🔍 SEARCH REQUEST:")
        print(f"   Query: '{query_text}'")
        print(f"   Results: {n_results}")
        if filter_category:
            print(f"   Filter: {filter_category}")
        
        # ----------------------------------------------------------------
        # STEP 2: Generate query embedding (YOUR Part 3 code!)
        # ----------------------------------------------------------------
        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True  # IMPORTANT: Same as Part 3
        )
        
        print(f"   ✓ Generated embedding: {query_embedding.shape}")
        
        # ----------------------------------------------------------------
        # STEP 3: Query ChromaDB (YOUR Part 4 code!)
        # ----------------------------------------------------------------
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results
        }
        
        # Add category filter if specified
        if filter_category:
            query_params["where"] = {"source_category": filter_category}
        
        results = collection.query(**query_params)
        
        print(f"   ✓ Found {len(results['documents'][0])} results")
        
        # ----------------------------------------------------------------
        # STEP 4: Format results for JSON response
        # ----------------------------------------------------------------
        formatted_results = []
        
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convert distance to similarity (0-1, higher is better)
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
        
        # ----------------------------------------------------------------
        # STEP 5: Return JSON response
        # ----------------------------------------------------------------
        response = {
            'query': query_text,
            'search_mode': 'semantic',
            'results': formatted_results,
            'count': len(formatted_results),
            'cached': False
        }

        set_cached_result(cache_key, response)
        print(f"   ✓ Cached result")

        return jsonify(response)
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/hybrid', methods=['POST'])
def hybrid_search():
    """
    Hybrid search: Combines semantic similarity with keyword matching.
    
    REQUEST:
    {
        "query": "neural networks optimization",
        "keywords": ["gradient", "descent"],
        "n_results": 5,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3
    }
    
    RESPONSE:
    {
        "results": [
            {
                "text": "...",
                "hybrid_score": 0.8234,
                "semantic_score": 0.7834,
                "keyword_score": 1.0
            }
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error':'Missing required field: query'}), 400
        
        query_text         = data['query']
        keywords           = data.get('keywords', [])
        n_results          = data.get('n_results', 5)
        filter_category    = data.get('filter_category', None)
        semantic_weight    = data.get('semantic_weight', 0.7)
        keyword_weight     = data.get('keyword_weight', 0.3)

        # Validate weights
        if not (0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1):
            return jsonify({'error': 'Weights must be between 0 and 1'}), 400
        
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        #CACHED CHECK (NEW)
        cache_key = generate_hybrid_cache_key(
            query_text,
            keywords,
            n_results,
            filter_category,
            semantic_weight,
            keyword_weight
        )
        cached_result = get_cached_result(cache_key)

        if cached_result is not None:
            print(f"\n🎯 CACHE HIT (HYBRID): '{query_text}' with {len(keywords)} keywords")
            cached_result['cached'] = True
            return jsonify(cached_result)
        

        print(f"\n❌ CACHE MISS (HYBRID): '{query_text}'")
        print(f"   Query: '{query_text}'")
        print(f"   Keywords: {keywords}")
        print(f"   Weights: {semantic_weight*100:.0f}% semantic + {keyword_weight*100:.0f}% keyword")
        
        print(f"\n🔍 HYBRID SEARCH REQUEST:")
        print(f"   Query: '{query_text}'")
        print(f"   Keywords: {keywords}")
        print(f"   Weights: {semantic_weight*100:.0f}% semantic + {keyword_weight*100:.0f}% keyword")

        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        retrieval_count = min(n_results*10, 100)

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results":retrieval_count
        }
        if filter_category:
            query_params["where"] = {"source_category":filter_category}

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
            print(f"   ✓ Returning top {len(top_results)} results")
             
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
            'cached': False 
            }
            
            set_cached_result(cache_key, response)
            print(f"   ✓ Cached result")
            return jsonify(response)
        

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache/stats', methods=['GET'])
def cache_statistics():
    """Get cache statistics"""
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
    """Handle 404 errors (endpoint not found)"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation',
        'available_endpoints': ['/', '/health', '/stats', '/search', '/hybrid']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors (internal server error)"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# Initialize the system (load model and database)
initialize_system()
    

if __name__ == '__main__':
    """
    This runs when you execute: python app.py
    
    Steps:
    1. Initialize system (load model + database)
    2. Start Flask server on port 5000
    3. Accept HTTP requests
    """
    
    print("\n" + "="*70)
    print("STARTING INTELLIGENT CONTENT RETRIEVAL SYSTEM API")
    print("="*70 + "\n")
    
    
    # Start the Flask server
    print("\n🌐 Starting Flask API server...")
    print("="*70)
    print("📍 API URL: http://localhost:5000")
    print("📚 Endpoints:")
    print("   • GET  /         - API information")
    print("   • GET  /health   - Health check")
    print("   • GET  /stats    - Database statistics")
    print("   • POST /search   - Semantic search")
    print("   • POST /hybrid   - Hybrid search")
    print("="*70)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    port = int(os.getenv('PORT', 5000))
    # Run Flask
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=port,        # Port number
        debug=False,        # Enable debug mode (shows detailed errors)
        threaded=True,        # ← ADD THIS for concurrent requests!
        use_reloader=False  
    )

"""
Flask API for Intelligent Content Retrieval System
PRODUCTION-READY VERSION WITH LAZY LOADING

Optimized for Render deployment with Gunicorn.
Works with both: python app.py AND gunicorn app:app

Author: Buhle Mlandu
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hashlib
import json
import time
from typing import Dict, List, Optional

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

def create_app():
    """
    Application factory pattern for Flask.
    Allows app to work with both direct execution and WSGI servers (Gunicorn).
    """
    app = Flask(__name__)
    CORS(app)
    
    # Configure app
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
    
    print("✓ Flask app created!")
    print("✓ CORS enabled")
    
    return app

app = create_app()

# ============================================================================
# LAZY LOADING GLOBALS
# ============================================================================
model = None
collection = None
model_name = 'all-mpnet-base-v2'

# Cache configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 100
result_cache: Dict[str, tuple] = {}

print("✓ Cache configured (TTL: 1 hour, Max: 100 queries)")


# ============================================================================
# LAZY LOADING FUNCTIONS
# ============================================================================

def load_model_lazy():
    """Lazy load the sentence transformer model (420MB)."""
    global model
    
    if model is not None:
        return model
    
    print("\n🔄 LAZY LOADING: Sentence Transformer Model...")
    print(f"   Model: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        
        print(f"   ✓ Model loaded: {model_name}")
        print(f"   ✓ Embedding dimensions: {embedding_dim}D")
        print(f"   ⚠️  Memory increased by ~420MB\n")
        
        return model
    
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise


def load_database_lazy():
    """Lazy load the ChromaDB database (~70MB)."""
    global collection
    
    if collection is not None:
        return collection
    
    print("\n🔄 LAZY LOADING: ChromaDB Database...")
    
    try:
        import chromadb
        
        # Try multiple possible paths (handles different deployment scenarios)
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'chromadb'),
            os.path.join(os.path.dirname(__file__), 'data', 'chromadb'),
            '/opt/render/project/src/data/chromadb',  # Render absolute path
            'data/chromadb'  # Relative path
        ]
        
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if not db_path:
            raise FileNotFoundError(
                f"Database not found! Tried paths:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="intelligent_content_retrieval")
        
        doc_count = collection.count()
        print(f"   ✓ Database loaded: {db_path}")
        print(f"   ✓ Collection: intelligent_content_retrieval")
        print(f"   ✓ Documents: {doc_count}")
        print(f"   ⚠️  Memory increased by ~70MB\n")
        
        return collection
    
    except Exception as e:
        print(f"   ❌ Failed to load database: {e}")
        raise


def ensure_system_loaded():
    """Ensure both model and database are loaded."""
    m = load_model_lazy()
    c = load_database_lazy()
    return m, c


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def generate_cache_key(query_text: str, n_results: int, filter_category: str = None) -> str:
    """Generate unique cache key from request parameters."""
    params = {
        'query': query_text,
        'n_results': n_results,
        'filter_category': filter_category
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
    keyword_weight: float = 0.3
) -> str:
    """Generate unique cache key for hybrid search."""
    params = {
        'query': query_text,
        'keywords': sorted(keywords) if keywords else [],
        'n_results': n_results,
        'filter_category': filter_category,
        'semantic_weight': semantic_weight,
        'keyword_weight': keyword_weight
    }
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    return cache_key


def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Get result from cache if it exists and is not expired."""
    if cache_key not in result_cache:
        return None
    
    result, timestamp = result_cache[cache_key]
    current_time = time.time()
    
    if current_time - timestamp > CACHE_TTL:
        del result_cache[cache_key]
        return None
    
    return result


def set_cached_result(cache_key: str, result: Dict) -> None:
    """Store result in cache with current timestamp."""
    global result_cache
    
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


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """Calculate keyword match score."""
    if not keywords or len(keywords) == 0:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword.lower().strip() in text_lower)
    return matches / len(keywords)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home page - provides information about the API"""
    return jsonify({
        'name': 'Intelligent Content Retrieval System API',
        'version': '2.1.0-production-ready',
        'author': 'Buhle Mlandu',
        'description': 'Production-ready semantic search API with lazy loading',
        'optimization': {
            'strategy': 'Lazy loading - Model and DB load on first request',
            'initial_memory': '~70MB (Flask + dependencies)',
            'peak_memory': '~560MB (after first search)',
            'memory_savings': '~510MB at startup (88% reduction)'
        },
        'deployment': {
            'platform': 'Render',
            'server': 'Gunicorn (recommended) or Flask dev server',
            'workers': 'Multi-worker support with Gunicorn'
        },
        'endpoints': {
            '/': 'API information (you are here)',
            '/health': 'Check system health',
            '/stats': 'Get database statistics',
            '/search': 'Semantic search (POST)',
            '/hybrid': 'Hybrid search (POST)',
            '/cache/stats': 'Cache statistics (GET)',
            '/cache/clear': 'Clear cache (POST)'
        },
        'status': 'operational',
        'loaded': {
            'model': model is not None,
            'database': collection is not None
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint (doesn't trigger loading).
    Perfect for Render's health checks and monitoring.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_loaded': collection is not None,
        'model_name': model_name,
        'embedding_dimensions': 768,
        'optimization': 'lazy-loading-enabled'
    })


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get detailed database statistics (triggers loading if needed)."""
    try:
        _, db = ensure_system_loaded()
        
        sample = db.peek(limit=50)
        
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
            'total_documents': db.count(),
            'model': model_name,
            'dimensions': 768,
            'distance_metric': 'cosine',
            'categories': categories,
            'system_loaded': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'system_loaded': False}), 500


@app.route('/search', methods=['POST'])
def semantic_search():
    """Perform semantic search (triggers loading on first call)."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query',
                'example': {'query': 'What is machine learning?', 'n_results': 5}
            }), 400
        
        query_text = data['query']
        n_results = data.get('n_results', 5)
        filter_category = data.get('filter_category', None)
        
        if not isinstance(n_results, int) or n_results < 1 or n_results > 20:
            return jsonify({'error': 'n_results must be between 1 and 20'}), 400
        
        # Check cache
        cache_key = generate_cache_key(query_text, n_results, filter_category)
        cached_result = get_cached_result(cache_key)
        
        if cached_result is not None:
            print(f"🎯 CACHE HIT: '{query_text[:50]}'")
            cached_result['cached'] = True
            return jsonify(cached_result)
        
        print(f"❌ CACHE MISS: '{query_text[:50]}'")
        
        # Lazy load
        m, db = ensure_system_loaded()
        
        # Generate embedding
        query_embedding = m.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Query
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results
        }
        
        if filter_category:
            query_params["where"] = {"source_category": filter_category}
        
        results = db.query(**query_params)
        
        # Format results
        formatted_results = []
        
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            
            formatted_results.append({
                'text': doc[:500] + "..." if len(doc) > 500 else doc,
                'similarity': round(similarity, 4),
                'metadata': {
                    'category': metadata.get('source_category', 'N/A'),
                    'url': metadata.get('source_url', 'N/A'),
                    'domain': metadata.get('source_domain', 'N/A'),
                    'chunk_index': metadata.get('chunk_index', 'N/A'),
                    'total_chunks': metadata.get('total_chunks_from_source', 'N/A')
                }
            })
        
        response = {
            'query': query_text,
            'search_mode': 'semantic',
            'results': formatted_results,
            'count': len(formatted_results),
            'cached': False
        }
        
        set_cached_result(cache_key, response)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/hybrid', methods=['POST'])
def hybrid_search():
    """Hybrid search (triggers loading on first call)."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query_text = data['query']
        keywords = data.get('keywords', [])
        n_results = data.get('n_results', 5)
        filter_category = data.get('filter_category', None)
        semantic_weight = data.get('semantic_weight', 0.7)
        keyword_weight = data.get('keyword_weight', 0.3)
        
        # Validate
        if not (0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1):
            return jsonify({'error': 'Weights must be between 0 and 1'}), 400
        
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400
        
        # Check cache
        cache_key = generate_hybrid_cache_key(
            query_text, keywords, n_results, filter_category,
            semantic_weight, keyword_weight
        )
        cached_result = get_cached_result(cache_key)
        
        if cached_result is not None:
            print(f"🎯 CACHE HIT (HYBRID): '{query_text[:50]}'")
            cached_result['cached'] = True
            return jsonify(cached_result)
        
        print(f"❌ CACHE MISS (HYBRID): '{query_text[:50]}'")
        
        # Lazy load
        m, db = ensure_system_loaded()
        
        # Generate embedding
        query_embedding = m.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Get candidates
        retrieval_count = min(n_results * 10, 100)
        
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": retrieval_count
        }
        
        if filter_category:
            query_params["where"] = {"source_category": filter_category}
        
        semantic_results = db.query(**query_params)
        
        # Calculate hybrid scores
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
                'text': doc[:500] + "..." if len(doc) > 500 else doc,
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
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/cache/stats', methods=['GET'])
def cache_statistics():
    """Get cache statistics"""
    stats = get_cache_stats()
    return jsonify({'cache_stats': stats, 'status': 'operational'})


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    """Clear all cached results"""
    entries_before = len(result_cache)
    clear_cache()
    return jsonify({
        'message': 'Cache cleared successfully',
        'entries_cleared': entries_before,
        'status': 'ok'
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/', '/health', '/stats', '/search', '/hybrid',
            '/cache/stats', '/cache/clear'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


# ============================================================================
# MAIN ENTRY POINT (for direct execution)
# ============================================================================

if __name__ == '__main__':
    """
    Direct execution with Flask development server.
    
    For PRODUCTION on Render, use Gunicorn instead:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
    """
    
    print("\n" + "="*70)
    print("INTELLIGENT CONTENT RETRIEVAL API - DEVELOPMENT MODE")
    print("="*70 + "\n")
    
    print("⚠️  WARNING: Using Flask development server")
    print("   For production, use Gunicorn instead\n")
    
    print("🚀 Memory Optimization: Lazy Loading")
    print("   Initial Memory: ~70MB")
    print("   Peak Memory: ~560MB (after first search)\n")
    
    print("🌐 Starting Flask API server...")
    print("="*70)
    print("📍 API URL: http://localhost:5000")
    print("📚 Endpoints: /, /health, /stats, /search, /hybrid")
    print("="*70)
    print("\n💡 First search request will be slower (loads model)")
    print("💡 Press CTRL+C to stop the server\n")
    
    port = int(os.getenv('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
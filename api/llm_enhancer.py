"""
LLM Enhancement Module
Rephrases search result chunks using Claude API for better readability
without changing factual content.

OPTIMIZATION: Batch processing + smart pre-enhancement for top 5 results

Author: Buhle Mlandu
"""

import os
from anthropic import Anthropic
from typing import List, Dict
import time

# Initialize Anthropic client
client = None

def initialize_claude():
    """Initialize Claude API client"""
    global client
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables!\n"
            "Please set it in your .env file or environment."
        )
    
    client = Anthropic(api_key=api_key)
    print("✓ Claude API client initialized")


def enhance_chunks_batch(chunks: List[str], user_query: str, max_retries: int = 2) -> Dict:
    """
    Enhance multiple chunks in a SINGLE API call (cost-efficient!).
    
    This is the KEY optimization - we send all chunks at once instead of
    making separate API calls for each chunk.
    
    Args:
        chunks: List of chunk texts to enhance
        user_query: The user's search query
        max_retries: Number of retry attempts if API fails
        
    Returns:
        dict: {
            'enhanced_chunks': List[str],  # Enhanced versions (same order)
            'success': bool,               # Whether enhancement succeeded
            'tokens_used': int,            # Total tokens consumed
            'chunks_processed': int        # Number of chunks processed
        }
    """
    
    if not client:
        return {
            'enhanced_chunks': chunks,  # Return originals
            'success': False,
            'tokens_used': 0,
            'chunks_processed': 0,
            'error': 'Claude client not initialized'
        }
    
    if not chunks or len(chunks) == 0:
        return {
            'enhanced_chunks': [],
            'success': True,
            'tokens_used': 0,
            'chunks_processed': 0
        }
    
    # STRICT SYSTEM PROMPT
    system_prompt = """You are a content reformatter. Your ONLY job is to rephrase text chunks to be more relevant and readable for a specific query.

CRITICAL RULES YOU MUST FOLLOW:
1. DO NOT add any information not present in the chunk
2. DO NOT remove any facts, statistics, or important details
3. DO NOT change the meaning or interpretation
4. DO NOT answer the query yourself - just rephrase the existing content
5. DO NOT add your own commentary or analysis
6. Keep the same approximate length (±20%)
7. Preserve all names, numbers, dates, and specific facts EXACTLY
8. If a chunk is already clear and well-phrased, return it unchanged

Your goal: Make each chunk easier to understand in the context of the query, while preserving ALL factual content."""

    # BUILD USER PROMPT - Format all chunks at once
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunks_text += f"\n--- CHUNK {i} ---\n{chunk}\n"
    
    user_prompt = f"""Query: "{user_query}"

{chunks_text}

Task: Rephrase each chunk above to be more directly relevant to the query while strictly following all rules. Return the rephrased chunks in the SAME ORDER, separated by "--- CHUNK N ---" headers.

If a chunk is already clear, return it unchanged. Maintain the same structure."""

    # Try with retries
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Fast Sonnet model
                max_tokens=4000,  # Enough for ~5 chunks
                temperature=0.3,  # Low temperature = faithful to source
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse the response back into individual chunks
            enhanced_chunks = []
            chunk_parts = response_text.split("--- CHUNK")
            
            for part in chunk_parts[1:]:  # Skip first empty part
                # Extract just the chunk text (remove "N ---" header)
                lines = part.split('\n', 1)
                if len(lines) > 1:
                    chunk_text = lines[1].strip()
                    # Remove trailing "--- CHUNK" if present
                    if chunk_text.endswith("---"):
                        chunk_text = chunk_text[:-3].strip()
                    enhanced_chunks.append(chunk_text)
            
            # Validation: Make sure we got the same number of chunks back
            if len(enhanced_chunks) != len(chunks):
                print(f"   ⚠️ Mismatch: sent {len(chunks)} chunks, got {len(enhanced_chunks)} back")
                # Fallback: pad with originals or truncate
                while len(enhanced_chunks) < len(chunks):
                    enhanced_chunks.append(chunks[len(enhanced_chunks)])
                enhanced_chunks = enhanced_chunks[:len(chunks)]
            
            return {
                'enhanced_chunks': enhanced_chunks,
                'success': True,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'chunks_processed': len(chunks)
            }
            
        except Exception as e:
            print(f"   ⚠️ Claude API error (attempt {attempt+1}/{max_retries+1}): {e}")
            
            if attempt < max_retries:
                time.sleep(1)  # Wait before retry
                continue
            else:
                # All retries failed - return originals
                return {
                    'enhanced_chunks': chunks,
                    'success': False,
                    'tokens_used': 0,
                    'chunks_processed': 0,
                    'error': str(e)
                }


def smart_enhance_results(results: List[Dict], user_query: str, auto_enhance_limit: int = 5) -> Dict:
    """
    SMART ENHANCEMENT: Only auto-enhance first N chunks (default: 5).
    
    This is your cost-saving strategy!
    
    Args:
        results: List of search results
        user_query: The original search query
        auto_enhance_limit: Number of chunks to auto-enhance (default: 5)
        
    Returns:
        dict: {
            'results': List[Dict],        # Results with enhancement data
            'auto_enhanced': int,         # Number auto-enhanced
            'pending_enhancement': int,   # Number NOT enhanced yet
            'total_tokens': int,          # Tokens used
            'cost_estimate': float        # Estimated cost in USD
        }
    """
    
    if not results or len(results) == 0:
        return {
            'results': [],
            'auto_enhanced': 0,
            'pending_enhancement': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0
        }
    
    total_chunks = len(results)
    chunks_to_enhance = min(total_chunks, auto_enhance_limit)
    
    print(f"\n🤖 Smart Enhancement Strategy:")
    print(f"   Total results: {total_chunks}")
    print(f"   Auto-enhancing: {chunks_to_enhance}")
    print(f"   Pending: {total_chunks - chunks_to_enhance}")
    
    # STEP 1: Extract first N chunks for batch processing
    chunks_to_process = [result['text'] for result in results[:chunks_to_enhance]]
    
    # STEP 2: Batch enhance (single API call!)
    enhancement_result = enhance_chunks_batch(chunks_to_process, user_query)
    
    # STEP 3: Update results with enhanced versions
    enhanced_results = []
    
    for i, result in enumerate(results):
        enhanced_result = result.copy()
        
        if i < chunks_to_enhance and enhancement_result['success']:
            # This chunk WAS enhanced
            enhanced_result['enhanced_text'] = enhancement_result['enhanced_chunks'][i]
            enhanced_result['enhancement_status'] = 'enhanced'
        else:
            # This chunk was NOT enhanced (beyond limit or API failed)
            enhanced_result['enhanced_text'] = None
            enhanced_result['enhancement_status'] = 'pending'
        
        enhanced_results.append(enhanced_result)
    
    # STEP 4: Calculate costs
    tokens_used = enhancement_result['tokens_used']
    # Claude Sonnet 4 pricing (approximate blended rate)
    cost_estimate = (tokens_used / 1_000_000) * 9.0
    
    print(f"   ✓ Enhanced {chunks_to_enhance} chunks in 1 API call")
    print(f"   ✓ Tokens used: {tokens_used:,}")
    print(f"   ✓ Cost: ${cost_estimate:.4f}")
    
    return {
        'results': enhanced_results,
        'auto_enhanced': chunks_to_enhance if enhancement_result['success'] else 0,
        'pending_enhancement': total_chunks - chunks_to_enhance,
        'total_tokens': tokens_used,
        'cost_estimate': cost_estimate
    }


def enhance_specific_chunks(results: List[Dict], chunk_indices: List[int], user_query: str) -> Dict:
    """
    Enhance SPECIFIC chunks by index (for on-demand enhancement).
    
    This is called when user clicks "Enhance remaining results" button.
    
    Args:
        results: Full list of results
        chunk_indices: List of indices to enhance (e.g., [5, 6, 7, 8, 9])
        user_query: The search query
        
    Returns:
        dict: {
            'results': List[Dict],     # Updated results
            'enhanced': int,           # Number newly enhanced
            'total_tokens': int,       # Tokens used
            'cost_estimate': float     # Cost in USD
        }
    """
    
    if not chunk_indices or len(chunk_indices) == 0:
        return {
            'results': results,
            'enhanced': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0
        }
    
    print(f"\n🎯 On-Demand Enhancement: {len(chunk_indices)} chunks")
    
    # Extract chunks to enhance
    chunks_to_process = [results[i]['text'] for i in chunk_indices if i < len(results)]
    
    # Batch enhance
    enhancement_result = enhance_chunks_batch(chunks_to_process, user_query)
    
    # Update results
    enhanced_results = results.copy()
    
    for idx, chunk_idx in enumerate(chunk_indices):
        if chunk_idx < len(enhanced_results) and enhancement_result['success']:
            enhanced_results[chunk_idx]['enhanced_text'] = enhancement_result['enhanced_chunks'][idx]
            enhanced_results[chunk_idx]['enhancement_status'] = 'enhanced'
    
    tokens_used = enhancement_result['tokens_used']
    cost_estimate = (tokens_used / 1_000_000) * 9.0
    
    print(f"   ✓ Enhanced {len(chunk_indices)} additional chunks")
    print(f"   ✓ Tokens: {tokens_used:,}")
    print(f"   ✓ Cost: ${cost_estimate:.4f}")
    
    return {
        'results': enhanced_results,
        'enhanced': len(chunk_indices) if enhancement_result['success'] else 0,
        'total_tokens': tokens_used,
        'cost_estimate': cost_estimate
    }
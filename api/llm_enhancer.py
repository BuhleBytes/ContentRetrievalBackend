"""
LLM Enhancement Module with Relevance Scoring

Rephrases search result chunks using Claude API for better readability
while preserving factual accuracy. Includes relevance assessment to help
users identify which results best answer their query.

Author: Buhle Mlandu
"""

import os
from anthropic import Anthropic
from typing import List, Dict
import time

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
    Enhance multiple chunks in a single API call with relevance scoring.
    
    Each chunk is rephrased for clarity and assessed for how well it answers
    the user's query. This batch approach is cost-efficient compared to
    making individual API calls for each chunk.
    
    Args:
        chunks: List of chunk texts to enhance
        user_query: The user's search query
        max_retries: Number of retry attempts if API fails
        
    Returns:
        dict: {
            'enhanced_chunks': List[Dict],
            'success': bool,
            'tokens_used': int,
            'chunks_processed': int
        }
    """
    
    if not client:
        return {
            'enhanced_chunks': [{'text': chunk, 'relevance': 'UNKNOWN'} for chunk in chunks],
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
    
    system_prompt = """You are a content reformatter with relevance assessment capabilities.

PRIMARY TASK: Rephrase text chunks to be more relevant and readable for a specific query.

CRITICAL RULES:
1. Never add information not present in the chunk
2. Never remove facts, statistics, or important details
3. Never change the meaning or interpretation
4. Never answer the query yourself, only rephrase existing content
5. Never add your own commentary or analysis
6. Keep the same approximate length (±20%)
7. Preserve all names, numbers, dates, and specific facts exactly
8. If a chunk is already clear and well-phrased, return it unchanged

RELEVANCE ASSESSMENT:
After rephrasing each chunk, assess how well it answers the user's query:

HIGH: Directly answers the query or provides critical information
MEDIUM: Provides related context or supporting information
LOW: Tangentially relevant or background information only

Format each chunk as:
REPHRASED: [your rephrased text]
RELEVANCE: [HIGH/MEDIUM/LOW]

Your goal is to make each chunk easier to understand in the context of the query while being completely faithful to the source."""

    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunks_text += f"\n--- CHUNK {i} ---\n{chunk}\n"
    
    user_prompt = f"""Query: "{user_query}"

{chunks_text}

Task: Rephrase each chunk above to be more relevant to the query, then assess its relevance level.

For each chunk, provide:
REPHRASED: [the rephrased text]
RELEVANCE: [HIGH/MEDIUM/LOW]

Maintain the "--- CHUNK N ---" structure in your response."""

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            enhanced_chunks = []
            chunk_parts = response_text.split("--- CHUNK")
            
            for part in chunk_parts[1:]:
                lines = part.split('\n')
                
                rephrased_text = ""
                relevance = "MEDIUM"
                
                capture_text = False
                for line in lines:
                    if line.strip().startswith("REPHRASED:"):
                        rephrased_text = line.replace("REPHRASED:", "").strip()
                        capture_text = True
                    elif line.strip().startswith("RELEVANCE:"):
                        relevance_raw = line.replace("RELEVANCE:", "").strip().upper()
                        if "HIGH" in relevance_raw:
                            relevance = "HIGH"
                        elif "LOW" in relevance_raw:
                            relevance = "LOW"
                        else:
                            relevance = "MEDIUM"
                        capture_text = False
                    elif capture_text and line.strip() and not line.strip().startswith("---"):
                        rephrased_text += " " + line.strip()
                
                if rephrased_text:
                    enhanced_chunks.append({
                        'text': rephrased_text.strip(),
                        'relevance': relevance
                    })
            
            if len(enhanced_chunks) != len(chunks):
                print(f"   ⚠️ Mismatch: sent {len(chunks)} chunks, got {len(enhanced_chunks)} back")
                while len(enhanced_chunks) < len(chunks):
                    enhanced_chunks.append({
                        'text': chunks[len(enhanced_chunks)],
                        'relevance': 'MEDIUM'
                    })
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
                time.sleep(1)
                continue
            else:
                return {
                    'enhanced_chunks': [{'text': chunk, 'relevance': 'UNKNOWN'} for chunk in chunks],
                    'success': False,
                    'tokens_used': 0,
                    'chunks_processed': 0,
                    'error': str(e)
                }


def smart_enhance_results(results: List[Dict], user_query: str, auto_enhance_limit: int = 5) -> Dict:
    """
    Smart enhancement strategy that only auto-enhances the first few chunks.
    
    This cost-saving approach processes the most relevant results automatically
    while leaving additional results available for on-demand enhancement.
    
    Args:
        results: List of search results
        user_query: The original search query
        auto_enhance_limit: Number of chunks to auto-enhance (default: 5)
        
    Returns:
        dict: {
            'results': List[Dict],
            'auto_enhanced': int,
            'pending_enhancement': int,
            'total_tokens': int,
            'cost_estimate': float,
            'relevance_summary': Dict
        }
    """
    
    if not results or len(results) == 0:
        return {
            'results': [],
            'auto_enhanced': 0,
            'pending_enhancement': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0,
            'relevance_summary': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        }
    
    total_chunks = len(results)
    chunks_to_enhance = min(total_chunks, auto_enhance_limit)
    
    print(f"\n🤖 Smart Enhancement Strategy:")
    print(f"   Total results: {total_chunks}")
    print(f"   Auto-enhancing: {chunks_to_enhance}")
    print(f"   Pending: {total_chunks - chunks_to_enhance}")
    
    chunks_to_process = [result['text'] for result in results[:chunks_to_enhance]]
    
    enhancement_result = enhance_chunks_batch(chunks_to_process, user_query)
    
    enhanced_results = []
    relevance_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
    
    for i, result in enumerate(results):
        enhanced_result = result.copy()
        
        if i < chunks_to_enhance and enhancement_result['success']:
            enhanced_chunk = enhancement_result['enhanced_chunks'][i]
            enhanced_result['enhanced_text'] = enhanced_chunk['text']
            enhanced_result['relevance'] = enhanced_chunk['relevance']
            enhanced_result['enhancement_status'] = 'enhanced'
            
            relevance_counts[enhanced_chunk['relevance']] += 1
        else:
            enhanced_result['enhanced_text'] = None
            enhanced_result['relevance'] = 'UNKNOWN'
            enhanced_result['enhancement_status'] = 'pending'
            relevance_counts['UNKNOWN'] += 1
        
        enhanced_results.append(enhanced_result)
    
    tokens_used = enhancement_result['tokens_used']
    cost_estimate = (tokens_used / 1_000_000) * 9.0
    
    print(f"   ✓ Enhanced {chunks_to_enhance} chunks in 1 API call")
    print(f"   ✓ Tokens used: {tokens_used:,}")
    print(f"   ✓ Cost: ${cost_estimate:.4f}")
    print(f"   ✓ Relevance: {relevance_counts['HIGH']} HIGH, {relevance_counts['MEDIUM']} MEDIUM, {relevance_counts['LOW']} LOW")
    
    return {
        'results': enhanced_results,
        'auto_enhanced': chunks_to_enhance if enhancement_result['success'] else 0,
        'pending_enhancement': total_chunks - chunks_to_enhance,
        'total_tokens': tokens_used,
        'cost_estimate': cost_estimate,
        'relevance_summary': relevance_counts
    }


def enhance_specific_chunks(results: List[Dict], chunk_indices: List[int], user_query: str) -> Dict:
    """
    Enhance specific chunks by index for on-demand enhancement.
    
    Called when user clicks a button to enhance results beyond the first five.
    
    Args:
        results: Full list of results
        chunk_indices: List of indices to enhance
        user_query: The search query
        
    Returns:
        dict: {
            'results': List[Dict],
            'enhanced': int,
            'total_tokens': int,
            'cost_estimate': float,
            'relevance_summary': Dict
        }
    """
    
    if not chunk_indices or len(chunk_indices) == 0:
        return {
            'results': results,
            'enhanced': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0,
            'relevance_summary': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        }
    
    print(f"\n🎯 On-Demand Enhancement: {len(chunk_indices)} chunks")
    
    chunks_to_process = [results[i]['text'] for i in chunk_indices if i < len(results)]
    
    enhancement_result = enhance_chunks_batch(chunks_to_process, user_query)
    
    enhanced_results = results.copy()
    relevance_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
    
    for idx, chunk_idx in enumerate(chunk_indices):
        if chunk_idx < len(enhanced_results) and enhancement_result['success']:
            enhanced_chunk = enhancement_result['enhanced_chunks'][idx]
            enhanced_results[chunk_idx]['enhanced_text'] = enhanced_chunk['text']
            enhanced_results[chunk_idx]['relevance'] = enhanced_chunk['relevance']
            enhanced_results[chunk_idx]['enhancement_status'] = 'enhanced'
            
            relevance_counts[enhanced_chunk['relevance']] += 1
    
    tokens_used = enhancement_result['tokens_used']
    cost_estimate = (tokens_used / 1_000_000) * 9.0
    
    print(f"   ✓ Enhanced {len(chunk_indices)} additional chunks")
    print(f"   ✓ Tokens: {tokens_used:,}")
    print(f"   ✓ Cost: ${cost_estimate:.4f}")
    print(f"   ✓ Relevance: {relevance_counts['HIGH']} HIGH, {relevance_counts['MEDIUM']} MEDIUM, {relevance_counts['LOW']} LOW")
    
    return {
        'results': enhanced_results,
        'enhanced': len(chunk_indices) if enhancement_result['success'] else 0,
        'total_tokens': tokens_used,
        'cost_estimate': cost_estimate,
        'relevance_summary': relevance_counts
    }
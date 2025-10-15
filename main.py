import asyncio
from datetime import datetime
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ollama import chat, AsyncClient
import requests
from os import getenv
from dotenv import load_dotenv
import trafilatura
from typing import AsyncGenerator, Dict, Any
from rank_bm25 import BM25Okapi
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
from aiohttp import ClientTimeout
import logging

app = FastAPI()

load_dotenv()

TOTAL_RESOURCE_TO_READ = int(getenv("TOP_RESOURCES_TO_READ", 5))
TOP_K_CHUNKS = int(getenv("TOP_K_CHUNKS", 3))
MAX_TOKEN_PER_CHUNK = int(getenv("MAX_TOKEN_PER_CHUNK", 100))
USE_COSINE_SIMILARITY = bool(int(getenv("USE_COSINE_SIMILARITY", 0)))
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIMENSIONS = int(getenv("EMBEDDING_DIMENSIONS", 256))
TEXT_GENERATION_MODEL = getenv("TEXT_GENERATION_MODEL", "gpt-oss:20b-cloud")

templates = Jinja2Templates(directory="templates")
templates.env.keep_trailing_newline = True

GOOGLE_API_KEY = getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CX = getenv("GOOGLE_CSE_ID")

nlp = spacy.load("en_core_web_sm")

session = aiohttp.ClientSession(timeout=ClientTimeout(total=10, connect=5))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def search(query: str) -> StreamingResponse:
    return StreamingResponse(answer_generator(query), media_type="text/event-stream")

async def web_search(query: str) -> list[dict]:
    """
    Perform a web search using Google Custom Search API
    
    Args:
        query: Search query string
        
    Returns:
        list: List of search result items, each containing title, link, and snippet
    """
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={query}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        search_results = []
        
        # Extract search results if they exist
        if 'items' in data:
            for item in data['items']:
                search_results.append({
                    'title': item.get('title', 'No title'),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', 'No description available')
                })
                
        return search_results
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request to Google Custom Search API: {e}")
        return []
    except (KeyError, ValueError) as e:
        logging.error(f"Error parsing API response: {e}")
        return []

async def answer_generator(query) -> AsyncGenerator[str, None]:
    search_results = await web_search(query)

    yield "event: search\ndata: " + json.dumps(search_results) + "\n\n"
    
    global session
    if not session:
        session = aiohttp.ClientSession(timeout=ClientTimeout(total=10, connect=5))

    fetch_start = time.perf_counter()
    web_fetch_tasks = [get_page_content(result) for result in search_results]
    await asyncio.gather(*web_fetch_tasks)
    fetch_duration = time.perf_counter() - fetch_start

    logging.info(f"Fetched {len(search_results)} pages in {fetch_duration:.4f}s")

    search_results = [result for result in search_results if result['content']][:TOTAL_RESOURCE_TO_READ]
    
    logging.info(f"Reading {len(search_results)} pages for context")

    # Create chunks
    chunk_start = time.perf_counter()
    for result in search_results:
        chunks = get_semantic_chunks(result['content'], max_tokens=MAX_TOKEN_PER_CHUNK)
        result['chunks'] = []
        for index, chunk in enumerate(chunks, start=1):
            result['chunks'].append({
                'id': index,
                'content': chunk
            })

    chunk_duration = time.perf_counter() - chunk_start
    logging.info(f"Created chunks in {chunk_duration:.4f}s")

    # Rank chunks using bm25
    bm25_start = time.perf_counter()
    query_tokens = query.lower().split()
    await asyncio.gather(*[
        score_result_chunks(result, query_tokens) 
        for result in search_results
    ])
    bm25_duration = time.perf_counter() - bm25_start
    logging.info(f"Calculated BM25 scores in {bm25_duration:.4f}s")

    if USE_COSINE_SIMILARITY:
        # Embedding
        query_embeddings = (await get_embeddings(query)).embeddings
        embedding_start = time.perf_counter()
        await asyncio.gather(*[
            generate_chunks_embeddings(result['chunks'])
            for result in search_results
        ])
        embedding_duration = time.perf_counter() - embedding_start
        logging.info(f"Generated Embeddings in {embedding_duration:.4f}s")

        # Calculate cosine similarity
        for result in search_results:
            chunk_embeddings = [chunk['embeddings'] for chunk in result['chunks']]
            cosine_scores = cosine_similarity(
                np.array(query_embeddings).reshape(1, -1), 
                np.array([emb for emb in chunk_embeddings])
            )

            for chunk, cosine_score in zip(result['chunks'], cosine_scores[0]):
                chunk['similarity_score'] = float(round(cosine_score, 8))
                

        # Reciprocal Rank Fusion
        k = 60
        for result in search_results:
            bm25_ranking = sorted(result['chunks'], key=lambda x: x["bm25_score"], reverse=True)
            emb_ranking = sorted(result['chunks'], key=lambda x: x["similarity_score"], reverse=True)

            bm25_ranks = build_rank_dict(bm25_ranking)
            emb_ranks = build_rank_dict(emb_ranking)
            
            for chunk in result['chunks']:
                rrf_score = (1 / (k + bm25_ranks[chunk["id"]])) + (1 / (k + emb_ranks[chunk["id"]]))
                chunk["rrf_score"] = rrf_score

        for result in search_results:
            result['chunks'].sort(key=lambda x: x['rrf_score'], reverse=True)

    else:
        # sort by bm25_score
        for result in search_results:
            result['chunks'].sort(key=lambda x: x['bm25_score'], reverse=True)

    context = ""
    for index, result in enumerate(search_results):
        context += f"\n\n[{index + 1}]\n\n"
        for chunk in result['chunks'][:TOP_K_CHUNKS]:
            context += f"\n\n{chunk['content']}\n\n"

    # Get the template first
    template = templates.get_template("prompts/prompt.jinja")

    # Then render it with context
    prompt = template.render(
        date=datetime.now().strftime("%d %b %Y"),
        context=context,
        query=query
    )

    logging.info('Prompt length - ' + str(len(prompt)))
    
    stream = chat(
        model=TEXT_GENERATION_MODEL,
        messages=[
            {
                'role': 'system', 
                'content': f"{prompt}"
            },
            {'role': 'user', 'content': query}
        ],
        stream=True,
        think=False,
        options={
            'num_ctx' : 32000,
            'num_predict': 5000,
            'temperature': 0.7,
            'repeat_penalty': 1.05
        }
    )

    for chunk in stream:
        if chunk['message']['content']:
            yield f"event: message\ndata: {json.dumps({ 'content': chunk['message']['content']})}\n\n"

def build_rank_dict(ranked_list: list[dict]) -> dict[int, int]:
    return {item["id"]: rank + 1 for rank, item in enumerate(ranked_list)}

async def get_page_content(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch and extract content from a URL with proper error handling and timeouts."""
    url: str = result.get('link', '')
    if not url:
        result['content'] = ""
        return result

    content = ""

    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                if html:
                    content = trafilatura.extract(
                        html, 
                        include_comments=False, 
                        include_tables=False
                    ) or ""
    except Exception as e:
        logging.warning(f"Error fetching {url}: {str(e)}")
    
    result['content'] = content

    result['content'] = content or ""

def get_semantic_chunks(text: str, max_tokens: int = 200, tokenizer=lambda s: s.split()) -> list[str]:
    doc = nlp(text)
    chunks, current_chunk, token_count = [], [], 0

    for sent in doc.sents:
        sent_tokens = len(tokenizer(sent.text))
        
        # If adding this sentence exceeds max_tokens, start a new chunk
        if token_count + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, token_count = [], 0
        
        current_chunk.append(sent.text)
        token_count += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

async def score_result_chunks(result: Dict[str, Any], query_tokens: list[str]) -> None:
    tokenized_chunks = [chunk['content'].lower().split() for chunk in result['chunks']]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query_tokens)
    
    for chunk, score in zip(result['chunks'], bm25_scores):
        chunk['bm25_score'] = float(round(score, 4))

async def get_embeddings(content: str) -> Any:
    return await AsyncClient().embed(
        model=EMBEDDING_MODEL, 
        dimensions=EMBEDDING_DIMENSIONS,
        input=content
    )

async def generate_chunks_embeddings(chunks: list[dict]) -> None:
    chunk_embeddings = await get_embeddings([chunk['content'] for chunk in chunks])
    for chunk, embedding in zip(chunks, chunk_embeddings.embeddings):
        chunk['embeddings'] = embedding
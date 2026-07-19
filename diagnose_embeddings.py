"""
Standalone diagnostic — NOT part of the app. Run this directly to see exactly
what happens when the embedding model loads, with real errors instead of a
silent hang inside FastAPI's threadpool.

Run with: venv\\Scripts\\python.exe diagnose_embeddings.py   (Windows)
       or: venv/bin/python diagnose_embeddings.py            (Mac/Linux)
"""
import time

print("Step 1: importing fastembed...")
t0 = time.time()
from langchain_community.embeddings import FastEmbedEmbeddings
print(f"  done in {time.time() - t0:.1f}s")

print("Step 2: initializing model (this downloads ~100-200MB on first run)...")
t0 = time.time()
try:
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print(f"  done in {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  FAILED after {time.time() - t0:.1f}s: {e}")
    raise SystemExit(1)

print("Step 3: embedding a test string...")
t0 = time.time()
vec = emb.embed_query("hello world")
print(f"  done in {time.time() - t0:.1f}s, got a vector of length {len(vec)}")

print("\nALL GOOD — the model works. If FastAPI still hangs, the problem is")
print("in how the server is calling this, not the model itself.")

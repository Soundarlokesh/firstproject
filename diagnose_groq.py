"""
Standalone diagnostic — NOT part of the app. Tests the Groq API call in
isolation, with a timeout, so a hang shows up as a clear error instead of
freezing forever like it does inside FastAPI's threadpool.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "")
print(f"Step 1: GROQ_API_KEY loaded, length={len(api_key)}, starts with '{api_key[:6]}...'")

if not api_key or api_key == "your_groq_api_key_here":
    print("  FAILED: key is empty or still the placeholder. Fix .env first.")
    raise SystemExit(1)

print("Step 2: importing langchain_groq...")
t0 = time.time()
from langchain_groq import ChatGroq
print(f"  done in {time.time() - t0:.1f}s")

print("Step 3: calling Groq API (10s timeout set)...")
t0 = time.time()
try:
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.4, timeout=10)
    response = llm.invoke([("human", "Say hello in one short sentence.")])
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  response: {response.content}")
except Exception as e:
    print(f"  FAILED after {time.time() - t0:.1f}s")
    print(f"  error type: {type(e).__name__}")
    print(f"  error: {e}")
    raise SystemExit(1)

print("\nALL GOOD — Groq API call works fine.")

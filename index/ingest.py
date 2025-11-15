# index/ingest.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from .vector_store import VectorStore

def load_txt_folder(folder: str):
  texts: List[str] = []
  metas: List[Dict[str, Any]] = []
  for root, _, files in os.walk(folder):
    for name in files:
      if name.lower().endswith(".txt"):
        path = os.path.join(root, name)
        with open(path, "r", encoding="utf-8") as f:
          for i, line in enumerate(f):
            line = line.strip()
            if not line:
              continue
            texts.append(line)
            metas.append({"source": path, "line": i + 1})
  return texts, metas

if __name__ == "__main__":
  folder = "data/processed"
  vs = VectorStore(path=".chroma", collection_name="docs")
  texts, metas = load_txt_folder(folder)
  print(f"loaded {len(texts)} chunks")
  if texts:
    vs.add_texts(texts, metas)
    print("index size:", vs.count())
  else:
    print("no .txt files found in", folder)

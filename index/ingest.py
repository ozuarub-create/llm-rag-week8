from __future__ import annotations

import os
import json
import hashlib
from typing import List, Dict, Any

from .vector_store import VectorStore


# =========================
# CONFIG
# =========================
INDEX_VERSION = "v1"
STATE_FILE = f"index_state_{INDEX_VERSION}.json"
CHROMA_PATH = f".chroma_{INDEX_VERSION}"
COLLECTION_NAME = "docs"


# =========================
# HELPERS
# =========================
def file_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_state() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =========================
# INGEST LOGIC
# =========================
def load_txt_folder(folder: str, state: Dict[str, Dict[str, Any]]):
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    new_state: Dict[str, Dict[str, Any]] = {}

    seen_docs = set()

    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue

            path = os.path.join(root, name)
            doc_id = path
            checksum = file_checksum(path)
            seen_docs.add(doc_id)

            prev = state.get(doc_id)

            # Skip unchanged, active documents
            if prev and prev["checksum"] == checksum and not prev.get("deleted", False):
                new_state[doc_id] = prev
                continue

            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    texts.append(line)
                    metas.append({
                        "doc_id": doc_id,
                        "checksum": checksum,
                        "deleted": False,
                        "source": path,
                        "line": i + 1,
                        "index_version": INDEX_VERSION
                    })

            new_state[doc_id] = {
                "checksum": checksum,
                "deleted": False
            }

    # Soft delete removed documents
    for doc_id, info in state.items():
        if doc_id not in seen_docs:
            new_state[doc_id] = {
                "checksum": info["checksum"],
                "deleted": True
            }

    return texts, metas, new_state


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    folder = "data/processed"

    vs = VectorStore(
        path=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    state = load_state()
    texts, metas, new_state = load_txt_folder(folder, state)

    print(f"[{INDEX_VERSION}] new / updated chunks: {len(texts)}")

    if texts:
        vs.add_texts(texts, metas)
        print("index updated")

    save_state(new_state)
    print("state saved")

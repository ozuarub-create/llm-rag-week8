# Week 8 – Incremental Indexing, Soft Delete, Index Versioning

This repository implements the Week 8 assignment for the RAG project.

It includes:

- **Incremental indexing**
  - Only new or modified documents are added.
  - Unchanged documents are skipped using stored checksums.

- **Soft delete**
  - Removed documents are marked as `"deleted": true` in state.
  - They are not re-indexed and can be filtered out at retrieval time.

- **Index versioning**
  - The index and state are stored under a version namespace (`v1`).
  - This allows rebuilds and rollback via changing `INDEX_VERSION`.

## How to run

1. Create a Python virtual environment and install requirements:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the incremental indexer:
```bash
python -m index.ingest
```

The index will be stored under `.chroma_v1/`, and state saved to `index_state_v1.json`.

## Adding documents

Add a new text file to:
```
data/processed/
```

Then run:
```bash
python -m index.ingest
```

Only the new document’s chunks will be indexed.

## Removing documents

Delete a file from:
```
data/processed/
```

Then run:
```bash
python -m index.ingest
```

The state file will mark it with `"deleted": true`.

## Notes

- The vector store is versioned by `INDEX_VERSION`.
- Always run ingestion with `python -m index.ingest`.

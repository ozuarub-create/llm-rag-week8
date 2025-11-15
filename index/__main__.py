# index/__main__.py
import argparse
import chromadb

from .vector_store import VectorStore
from .ingest import load_txt_folder


def build_index(src_folder: str, db_path: str = ".chroma", collection: str = "docs", reset: bool = False):
    if reset:
        # wipe old collection if it exists
        client = chromadb.PersistentClient(path=db_path)
        try:
            client.delete_collection(name=collection)
            print(f"reset: deleted old collection '{collection}'")
        except Exception:
            # ok if it didn't exist
            pass

    vs = VectorStore(path=db_path, collection_name=collection)
    texts, metas = load_txt_folder(src_folder)
    print(f"loaded {len(texts)} chunks from {src_folder}")
    if texts:
        vs.add_texts(texts, metas)
    print("index size:", vs.count())


def main():
    parser = argparse.ArgumentParser(prog="index", description="Simple index CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the vector index from a folder of .txt files")
    p_build.add_argument("--from", dest="src_folder", required=True, help="Path to folder with .txt files")
    p_build.add_argument("--db-path", default=".chroma", help="Chroma database path (folder)")
    p_build.add_argument("--collection", default="docs", help="Collection name")
    p_build.add_argument("--reset", action="store_true", help="Delete the collection before building")
    p_query = sub.add_parser("query", help="Run a search query against the vector store")
    p_query.add_argument("question", help="The question or text to search for")
    p_query.add_argument("--k", type=int, default=3, help="Number of results to return")
    p_query.add_argument("--db-path", default=".chroma", help="Chroma database path")
    p_query.add_argument("--collection", default="docs", help="Collection name")
    args = parser.parse_args()

    if args.cmd == "build":
        build_index(args.src_folder, db_path=args.db_path, collection=args.collection, reset=args.reset)
    elif args.cmd == "query":
        query_index(args.question, k=args.k, db_path=args.db_path, collection=args.collection)
   

def query_index(question: str, k: int = 3, db_path: str = ".chroma", collection: str = "docs"):
    from .vector_store import VectorStore
    vs = VectorStore(path=db_path, collection_name=collection)
    results = vs.similarity_search(question, k=k)
    if not results:
        print("No results found.")
        return
    for r in results:
        print(f"{r['distance']:.4f} :: {r['text']} | {r['metadata']}")

if __name__ == "__main__":
    main()

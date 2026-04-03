"""
create_memory_for_llm.py
────────────────────────
Standalone script to (re)build the FAISS vector store from all PDFs in data/.

Usage:
    python create_memory_for_llm.py

All tuneable parameters (chunk size, overlap, paths, embedding model) are
controlled via config.py.  The heavy lifting is done by services/ingestion.py
so this script stays in sync with the in-app rebuild logic automatically.
"""

from pathlib import Path
from config import DATA_PATH, DB_FAISS_PATH
from services import ingestion

def main():
    data_dir = Path(DATA_PATH)
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{DATA_PATH}'. Add PDFs and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")
    print("Building vector store (force_rebuild=True)…")

    result = ingestion.build_vectorstore(pdf_files, force_rebuild=True)

    print(f"\nDone.")
    print(f"  Indexed : {result['indexed']}")
    print(f"  Skipped : {result['skipped']}")
    print(f"  Chunks  : {result['total_new_chunks']}")
    print(f"\nVector store saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()

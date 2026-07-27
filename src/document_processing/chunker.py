"""
Chunking Strategy
-----------------
Recursive/character-window chunking with overlap:
  - CHUNK_SIZE   ~800-1000 characters (default 1000)
  - CHUNK_OVERLAP ~100-150 characters (default 150)

Justification (see README "Design Decisions" for the full write-up):
  1. 800-1000 characters (~150-220 tokens) is small enough to keep each
     chunk semantically focused (one embedding represents one idea) but
     large enough to retain surrounding context for the LLM to reason over.
  2. A 100-150 character overlap prevents sentences/ideas that straddle a
     chunk boundary from being split apart and losing meaning — without the
     overlap, a fact stated at the end of chunk N might be truncated and
     become unretrievable/unusable in isolation.
  3. Splitting is attempted first on paragraph boundaries, then sentence
     boundaries, and only falls back to a hard character cut when no
     natural boundary exists nearby — this keeps chunks readable rather
     than cutting mid-word.
  4. Page-number metadata is preserved per chunk so every retrieved chunk
     can still be cited back to an exact document + page.
"""
from typing import List, Dict, Any

SEPARATORS = ["\n\n", "\n", ". ", " "]


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Recursively splits text using a priority list of separators, falling
        back to a hard character window when no separator keeps chunks small
        enough."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                chunks: List[str] = []
                current = ""
                for part in parts:
                    candidate = (current + sep + part) if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        # If a single part is still too big, recurse with the next separator
                        if len(part) > self.chunk_size:
                            chunks.extend(self._split_text(part))
                            current = ""
                        else:
                            current = part
                if current:
                    chunks.append(current)
                return self._apply_overlap(chunks)

        # No separator found at all -> hard character window
        return self._apply_overlap(
            [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        )

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Prepends the tail of the previous chunk to each subsequent chunk so
        context flows across boundaries."""
        if len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-self.chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits each page's text into overlapping chunks while preserving
        doc_id / page_number metadata for citation."""
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        for page in pages_data:
            for piece in self._split_text(page["text"]):
                if not piece.strip():
                    continue
                chunks.append({
                    "chunk_id": f"{page['doc_id']}_c{chunk_counter}",
                    "doc_id": page["doc_id"],
                    "page_number": page["page_number"],
                    "text": piece,
                    "char_count": len(piece),
                })
                chunk_counter += 1

        return chunks

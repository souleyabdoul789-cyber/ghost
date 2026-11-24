"""ghost.memory — mémoire documentaire persistante sans dépendance externe.

Les connaissances apprises sur le Web restent sourcées et peuvent être
recherchées sans modifier immédiatement les poids du modèle.
"""

from dataclasses import dataclass
import hashlib
import math
import os
import re
import sqlite3
import time

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class MemoryHit:
    document_id: str
    title: str
    source: str
    content: str
    score: float
    chunk_index: int


class GMemory:
    """Index local SQLite des documents, passages et retours validés.

    Le moteur est volontairement lexical (BM25 léger) : il ne prétend pas
    faire de recherche sémantique par embeddings. Il fonctionne hors ligne
    et ne requiert aucune base vectorielle externe.
    """

    def __init__(self, path="g_memory.sqlite3", chunk_chars=1200):
        self.path = os.fspath(path)
        self.chunk_chars = max(200, int(chunk_chars))
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        """)
        self._db.commit()

    @staticmethod
    def _tokens(text):
        return [token.lower() for token in _WORD_RE.findall(text or "")]

    def _split(self, text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        chunks, current = [], ""
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_chars:
                if current:
                    chunks.append(current)
                    current = ""
                start = 0
                while start < len(paragraph):
                    end = min(len(paragraph), start + self.chunk_chars)
                    if end < len(paragraph):
                        boundary = paragraph.rfind(" ", start, end)
                        if boundary > start + self.chunk_chars // 2:
                            end = boundary
                    chunks.append(paragraph[start:end].strip())
                    start = end
                continue
            candidate = (current + "\n\n" + paragraph).strip() if current else paragraph
            if len(candidate) > self.chunk_chars:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or ([text.strip()] if (text or "").strip() else [])

    def add_document(self, document):
        """Ajoute un WebDocument ; renvoie False si son contenu existe déjà."""
        content = (document.text or "").strip()
        if not content:
            return False
        digest = document.sha256 or hashlib.sha256(content.encode("utf-8")).hexdigest()
        doc_id = digest[:32]
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO documents(id,url,title,content,sha256,source_type,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (doc_id, document.url, document.title or document.url, content,
             digest, "web", float(document.fetched_at or time.time())),
        )
        if cursor.rowcount == 0:
            return False
        for index, chunk in enumerate(self._split(content)):
            self._db.execute(
                "INSERT INTO chunks(document_id,chunk_index,content,token_count) VALUES(?,?,?,?)",
                (doc_id, index, chunk, len(self._tokens(chunk))),
            )
        self._db.commit()
        return True

    def add_note(self, text, title="G validated note", source="user-feedback"):
        """Enregistre une correction ou une connaissance validée par l'utilisateur."""
        content = (text or "").strip()
        if not content:
            raise ValueError("La note ne peut pas être vide.")
        digest = hashlib.sha256((source + "\0" + title + "\0" + content).encode("utf-8")).hexdigest()
        doc_id = digest[:32]
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO documents(id,url,title,content,sha256,source_type,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (doc_id, source, title, content, digest, "validated-note", time.time()),
        )
        if cursor.rowcount == 0:
            return False
        for index, chunk in enumerate(self._split(content)):
            self._db.execute(
                "INSERT INTO chunks(document_id,chunk_index,content,token_count) VALUES(?,?,?,?)",
                (doc_id, index, chunk, len(self._tokens(chunk))),
            )
        self._db.commit()
        return True

    def search(self, query, limit=5):
        terms = self._tokens(query)
        if not terms:
            return []
        rows = self._db.execute(
            "SELECT c.document_id,c.chunk_index,c.content,c.token_count,d.title,d.url "
            "FROM chunks c JOIN documents d ON d.id=c.document_id"
        ).fetchall()
        if not rows:
            return []
        tokenized = [self._tokens(row[2]) for row in rows]
        document_frequency = {}
        for tokens in tokenized:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        avgdl = sum(max(1, row[3]) for row in rows) / len(rows)
        total = len(rows)
        hits = []
        for row, tokens in zip(rows, tokenized):
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            length = max(1, row[3])
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * length / max(avgdl, 1.0))
                score += idf * (tf * 2.2) / denominator
            if score > 0:
                hits.append(MemoryHit(row[0], row[4], row[5], row[2], score, row[1]))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:max(1, int(limit))]

    def documents(self):
        return self._db.execute(
            "SELECT id,url,title,sha256,source_type,created_at FROM documents ORDER BY created_at"
        ).fetchall()

    def export_corpus(self, path):
        """Exporte les documents mémorisés en texte brut pour ré-entraînement contrôlé."""
        rows = self._db.execute("SELECT title,url,content FROM documents ORDER BY created_at").fetchall()
        temporary = os.fspath(path) + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            for title, url, content in rows:
                stream.write("\n\n### " + title + "\nSource: " + url + "\n\n")
                stream.write(content)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return len(rows)

    def close(self):
        if getattr(self, "_db", None) is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
